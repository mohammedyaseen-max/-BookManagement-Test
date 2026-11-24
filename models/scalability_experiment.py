from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import json
import os
import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("BookCrossing-Scalability-Experiment")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

INPUT = str(ROOT / "data" / "processed" / "ratings_enriched")
OUTPUT = str(ROOT / "analysis" / "results" / "scalability_experiment")
os.makedirs(OUTPUT, exist_ok=True)

# Reuse the explicit-rating task and same feature design as the RF comparison.
data = (
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
total = data.count()
if total == 0:
    raise ValueError(f"No explicit rating records found in {INPUT}")

rmse_eval = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="mae")

results = []
for fraction, label in [(0.25, "25_percent"), (0.50, "50_percent"), (1.00, "100_percent")]:
    # Fixed seed; sample only the input rows, then split sampled rows into train/test.
    sampled = data.sample(withReplacement=False, fraction=fraction, seed=2026).cache()
    sampled_count = sampled.count()
    if sampled_count < 20:
        sampled.unpersist()
        continue

    train, test = sampled.randomSplit([0.8, 0.2], seed=42)
    train = train.cache()
    test = test.cache()
    train_count = train.count()
    test_count = test.count()

    if train_count == 0 or test_count == 0:
        train.unpersist()
        test.unpersist()
        sampled.unpersist()
        continue

    global_mean = float(train.agg(F.avg("rating")).first()[0])

    # Aggregate features are calculated from training data only.
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
              .withColumn("user_avg_rating", F.coalesce(F.col("user_avg_rating"), F.lit(global_mean)))
              .withColumn("user_rating_count", F.coalesce(F.col("user_rating_count"), F.lit(0.0)))
              .withColumn("book_avg_rating", F.coalesce(F.col("book_avg_rating"), F.lit(global_mean)))
              .withColumn("book_rating_count", F.coalesce(F.col("book_rating_count"), F.lit(0.0)))
        )

    features = ["user_avg_rating", "user_rating_count", "book_avg_rating", "book_rating_count"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    # Measure end-to-end pipeline runtime (feature aggregation + model fit + test prediction).
    start = time.perf_counter()
    train_features = assembler.transform(add_features(train)).cache()
    test_features = assembler.transform(add_features(test)).cache()

    # Materialize feature engineering before model fitting so the timing stages are clear.
    train_features.count()
    test_features.count()
    feature_seconds = round(time.perf_counter() - start, 2)

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="rating",
        predictionCol="prediction",
        numTrees=40,
        maxDepth=8,
        seed=42
    )

    fit_start = time.perf_counter()
    model = rf.fit(train_features)
    fit_seconds = round(time.perf_counter() - fit_start, 2)

    predict_start = time.perf_counter()
    predictions = model.transform(test_features).cache()
    pred_count = predictions.count()
    predict_seconds = round(time.perf_counter() - predict_start, 2)

    rmse = float(rmse_eval.evaluate(predictions))
    mae = float(mae_eval.evaluate(predictions))

    results.append({
        "dataset_fraction": fraction,
        "dataset_label": label,
        "input_rows": sampled_count,
        "train_rows": train_count,
        "test_rows": test_count,
        "prediction_rows": pred_count,
        "coverage": pred_count / test_count if test_count else 0.0,
        "feature_engineering_seconds": feature_seconds,
        "training_seconds": fit_seconds,
        "prediction_seconds": predict_seconds,
        "total_pipeline_seconds": round(feature_seconds + fit_seconds + predict_seconds, 2),
        "rmse": rmse,
        "mae": mae,
        "algorithm": "Spark MLlib RandomForestRegressor",
        "numTrees": 40,
        "maxDepth": 8,
        "seed_sample": 2026,
        "seed_split": 42
    })

    print(f"\n=== {label} ===")
    print(json.dumps(results[-1], indent=2))

    predictions.unpersist()
    train_features.unpersist()
    test_features.unpersist()
    train.unpersist()
    test.unpersist()
    sampled.unpersist()

timestamp = datetime.now(timezone.utc).isoformat()
payload = {
    "experiment": "Single-machine data-volume scalability experiment",
    "input_parquet": INPUT,
    "available_explicit_interactions": total,
    "rating_filter": "Book-Rating 1 through 10; rating 0 excluded",
    "environment": {
        "spark_master": "local[*]",
        "spark_driver_memory": "2g",
        "algorithm": "Spark MLlib RandomForestRegressor",
        "numTrees": 40,
        "maxDepth": 8,
        "features": [
            "user_avg_rating", "user_rating_count",
            "book_avg_rating", "book_rating_count"
        ],
        "feature_statistics_source": "training data only",
        "sample_seed": 2026,
        "train_test_split": "80/20 randomSplit, seed=42",
        "measurement_note": "Times are machine- and environment-specific; rerun if you need repeated-run averages."
    },
    "created_utc": timestamp,
    "results": results
}

json_path = os.path.join(OUTPUT, "scalability_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

# Also write a CSV for easy plotting and report tables, without requiring pandas.
csv_path = os.path.join(OUTPUT, "scalability_results.csv")
if results:
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

print("\n=== SCALABILITY EXPERIMENT COMPLETE ===")
print(json.dumps(payload, indent=2))
print(f"\nSaved JSON: {json_path}")
print(f"Saved CSV:  {csv_path}")

data.unpersist()
spark.stop()
