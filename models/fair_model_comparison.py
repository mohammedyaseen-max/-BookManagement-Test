import json
import os
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("BookCrossing-Fair-Model-Comparison")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

INPUT = "data/processed/ratings_enriched"
OUTPUT = "analysis/results/fair_model_comparison"
os.makedirs(OUTPUT, exist_ok=True)

# Load the same explicit-rating dataset for every model.
raw = (
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
usable_rows = raw.count()

# Create an integer item ID mapping. The mapping uses identifiers only, not ratings.
indexer = StringIndexer(inputCol="isbn", outputCol="item_id", handleInvalid="skip")
indexed = indexer.fit(raw).transform(raw).withColumn("item_id", F.col("item_id").cast("int"))
data = indexed.select("user_id", "isbn", "item_id", "rating").cache()

# One shared, deterministic split for ALS, Random Forest, and the baseline.
train, test = data.randomSplit([0.8, 0.2], seed=42)
train = train.cache()
test = test.cache()
train_count = train.count()
test_count = test.count()

if train_count == 0 or test_count == 0:
    raise ValueError(f"Empty split: train={train_count}, test={test_count}")

rmse_eval = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="rmse"
)
mae_eval = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="mae"
)

global_mean = float(train.agg(F.avg("rating")).first()[0])
baseline_all = test.withColumn("prediction", F.lit(global_mean))
baseline_all_rmse = float(rmse_eval.evaluate(baseline_all))
baseline_all_mae = float(mae_eval.evaluate(baseline_all))

# Model 1: ALS collaborative filtering.
als_cfg = {"rank": 10, "regParam": 0.1, "maxIter": 10}
als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    rank=als_cfg["rank"],
    regParam=als_cfg["regParam"],
    maxIter=als_cfg["maxIter"],
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)
als_start = time.time()
als_model = als.fit(train.select("user_id", "item_id", "rating"))
als_train_seconds = round(time.time() - als_start, 2)
als_predictions = als_model.transform(test).cache()
als_pred_count = als_predictions.count()
als_rmse = float(rmse_eval.evaluate(als_predictions)) if als_pred_count else None
als_mae = float(mae_eval.evaluate(als_predictions)) if als_pred_count else None

# Build a strict common evaluation subset where ALS can make a prediction.
als_valid_keys = als_predictions.select("user_id", "isbn", "rating", "prediction")
common_count = als_valid_keys.count()

# Baseline on exactly the same rows as ALS.
baseline_common = als_valid_keys.withColumn("prediction", F.lit(global_mean))
baseline_common_rmse = float(rmse_eval.evaluate(baseline_common)) if common_count else None
baseline_common_mae = float(mae_eval.evaluate(baseline_common)) if common_count else None

# Model 2: Random Forest. User/book aggregate features come from training rows only.
user_stats = train.groupBy("user_id").agg(
    F.avg("rating").alias("user_avg_rating"),
    F.count("*").cast("double").alias("user_rating_count")
)
book_stats = train.groupBy("isbn").agg(
    F.avg("rating").alias("book_avg_rating"),
    F.count("*").cast("double").alias("book_rating_count")
)

def add_features(df):
    return (
        df.join(user_stats, "user_id", "left")
          .join(book_stats, "isbn", "left")
          .withColumn("user_avg_rating", F.coalesce("user_avg_rating", F.lit(global_mean)))
          .withColumn("user_rating_count", F.coalesce("user_rating_count", F.lit(0.0)))
          .withColumn("book_avg_rating", F.coalesce("book_avg_rating", F.lit(global_mean)))
          .withColumn("book_rating_count", F.coalesce("book_rating_count", F.lit(0.0)))
    )

train_rf = add_features(train).cache()
test_rf = add_features(test).cache()
feature_cols = [
    "user_avg_rating", "user_rating_count",
    "book_avg_rating", "book_rating_count"
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

rf_start = time.time()
rf_model = rf.fit(assembler.transform(train_rf))
rf_train_seconds = round(time.time() - rf_start, 2)
rf_predictions = rf_model.transform(assembler.transform(test_rf)).cache()
rf_pred_count = rf_predictions.count()
rf_all_rmse = float(rmse_eval.evaluate(rf_predictions))
rf_all_mae = float(mae_eval.evaluate(rf_predictions))

# Evaluate RF on the exact ALS-evaluable test rows as well.
common_keys = als_valid_keys.select("user_id", "isbn", "rating")
rf_common = (
    rf_predictions.select("user_id", "isbn", "rating", "prediction")
    .join(common_keys, ["user_id", "isbn", "rating"], "inner")
    .cache()
)
rf_common_count = rf_common.count()
rf_common_rmse = float(rmse_eval.evaluate(rf_common)) if rf_common_count else None
rf_common_mae = float(mae_eval.evaluate(rf_common)) if rf_common_count else None

metrics = {
    "task": "Fair comparison of explicit book-rating prediction models",
    "input_parquet": INPUT,
    "rating_filter": "Book-Rating 1 through 10; rating 0 excluded",
    "usable_interactions": usable_rows,
    "split": {"method": "randomSplit", "fractions": [0.8, 0.2], "seed": 42},
    "train_rows": train_count,
    "test_rows": test_count,
    "global_mean_rating_from_train": global_mean,
    "all_test_rows": {
        "global_mean_baseline": {
            "rmse": baseline_all_rmse, "mae": baseline_all_mae,
            "evaluated_rows": test_count
        },
        "random_forest": {
            "numTrees": 40, "maxDepth": 8,
            "features": feature_cols,
            "rmse": rf_all_rmse, "mae": rf_all_mae,
            "evaluated_rows": rf_pred_count,
            "coverage": rf_pred_count / test_count if test_count else 0,
            "training_seconds": rf_train_seconds
        },
        "als": {
            **als_cfg,
            "rmse": als_rmse, "mae": als_mae,
            "evaluated_rows": als_pred_count,
            "coverage": als_pred_count / test_count if test_count else 0,
            "training_seconds": als_train_seconds
        }
    },
    "same_test_rows_comparison": {
        "description": "All three methods are scored on the subset where ALS returns a prediction.",
        "common_rows": common_count,
        "global_mean_baseline": {
            "rmse": baseline_common_rmse, "mae": baseline_common_mae
        },
        "random_forest": {
            "rmse": rf_common_rmse, "mae": rf_common_mae,
            "evaluated_rows": rf_common_count
        },
        "als": {
            "rmse": als_rmse, "mae": als_mae,
            "evaluated_rows": als_pred_count
        }
    },
    "notes": [
        "Lower RMSE and MAE are better.",
        "ALS coverage is lower when test users or items were not represented in training.",
        "Random Forest aggregate features are computed from training data only.",
        "Training times are specific to this machine, Docker image, and current local Spark configuration."
    ]
}

with open(os.path.join(OUTPUT, "comparison_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n=== FAIR MODEL COMPARISON ===")
print(json.dumps(metrics, indent=2))
print(f"\nResults saved to: {OUTPUT}/comparison_metrics.json")

rf_predictions.unpersist()
rf_common.unpersist()
als_predictions.unpersist()
train_rf.unpersist()
test_rf.unpersist()
train.unpersist()
test.unpersist()
data.unpersist()
raw.unpersist()
spark.stop()
