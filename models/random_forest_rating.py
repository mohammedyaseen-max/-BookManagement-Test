from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import json
import os
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("BookCrossing-RandomForest-Rating")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

INPUT = str(ROOT / "data" / "processed" / "ratings_enriched")
OUTPUT = str(ROOT / "analysis" / "results" / "random_forest_rating")
os.makedirs(OUTPUT, exist_ok=True)

# Load explicit ratings and valid numeric user IDs.
ratings = (
    spark.read.parquet(INPUT)
    .select(
        F.col("User-ID").cast("int").alias("user_id"),
        F.trim(F.col("ISBN")).alias("isbn"),
        F.col("Book-Rating").cast("double").alias("rating")
    )
    .filter(
        F.col("user_id").isNotNull()
        & F.col("isbn").isNotNull()
        & (F.col("isbn") != "")
        & F.col("rating").between(1, 10)
    )
    .cache()
)

total_rows = ratings.count()
train_raw, test_raw = ratings.randomSplit([0.8, 0.2], seed=42)
train_raw = train_raw.cache()
test_raw = test_raw.cache()
train_count = train_raw.count()
test_count = test_raw.count()

if train_count == 0 or test_count == 0:
    raise ValueError(f"Empty split: train={train_count}, test={test_count}")

# Compute user and book aggregates from TRAINING DATA ONLY to avoid target leakage.
user_stats = train_raw.groupBy("user_id").agg(
    F.avg("rating").alias("user_avg_rating"),
    F.count("*").cast("double").alias("user_rating_count")
)
book_stats = train_raw.groupBy("isbn").agg(
    F.avg("rating").alias("book_avg_rating"),
    F.count("*").cast("double").alias("book_rating_count")
)
global_mean = float(train_raw.agg(F.avg("rating")).first()[0])

def add_features(df):
    return (
        df.join(user_stats, "user_id", "left")
          .join(book_stats, "isbn", "left")
          .withColumn("user_avg_rating", F.coalesce("user_avg_rating", F.lit(global_mean)))
          .withColumn("user_rating_count", F.coalesce("user_rating_count", F.lit(0.0)))
          .withColumn("book_avg_rating", F.coalesce("book_avg_rating", F.lit(global_mean)))
          .withColumn("book_rating_count", F.coalesce("book_rating_count", F.lit(0.0)))
    )

train = add_features(train_raw).cache()
test = add_features(test_raw).cache()

# Baseline on the exact same test set.
evaluator_rmse = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="mae"
)
baseline = test.withColumn("prediction", F.lit(global_mean))
baseline_rmse = float(evaluator_rmse.evaluate(baseline))
baseline_mae = float(evaluator_mae.evaluate(baseline))

feature_cols = [
    "user_avg_rating",
    "user_rating_count",
    "book_avg_rating",
    "book_rating_count"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="rating",
    predictionCol="prediction",
    numTrees=40,
    maxDepth=8,
    seed=42
)
pipeline = Pipeline(stages=[assembler, rf])

start = time.time()
model = pipeline.fit(train)
training_seconds = round(time.time() - start, 2)

predictions = model.transform(test).cache()
prediction_count = predictions.count()
rf_rmse = float(evaluator_rmse.evaluate(predictions))
rf_mae = float(evaluator_mae.evaluate(predictions))

metrics = {
    "task": "Explicit book-rating prediction using Spark MLlib Random Forest",
    "input_parquet": INPUT,
    "rating_filter": "Book-Rating 1 through 10; rating 0 excluded",
    "feature_engineering": "User/book aggregates computed from training split only",
    "total_usable_interactions": total_rows,
    "train_rows": train_count,
    "test_rows": test_count,
    "prediction_rows": prediction_count,
    "coverage": prediction_count / test_count if test_count else 0.0,
    "global_mean_rating_from_train": global_mean,
    "random_forest": {
        "numTrees": 40,
        "maxDepth": 8,
        "features": feature_cols,
        "rmse": rf_rmse,
        "mae": rf_mae,
        "training_seconds": training_seconds
    },
    "global_mean_baseline_same_test": {
        "rmse": baseline_rmse,
        "mae": baseline_mae
    },
    "note": "Lower RMSE and MAE are better. Feature aggregates are calculated using training data only."
}

with open(os.path.join(OUTPUT, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Save a small sample for inspection, not the entire prediction dataset.
(
    predictions.select("user_id", "isbn", "rating", "prediction")
    .orderBy(F.rand(seed=42))
    .limit(100)
    .coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv(os.path.join(OUTPUT, "sample_predictions"))
)

print("\n=== RANDOM FOREST MODEL EVALUATION ===")
print(json.dumps(metrics, indent=2))
print(f"\nOutputs saved to: {OUTPUT}")

predictions.unpersist()
train.unpersist()
test.unpersist()
train_raw.unpersist()
test_raw.unpersist()
ratings.unpersist()
spark.stop()
