from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import csv
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
    .appName("BookCrossing-Scalability-Repeated-Runs")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

INPUT = str(ROOT / "data" / "processed" / "ratings_enriched")
OUTPUT = str(ROOT / "analysis" / "results" / "scalability_experiment_repeated")
REPEATS = 3
os.makedirs(OUTPUT, exist_ok=True)

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
configs = [(0.25, "25_percent"), (0.50, "50_percent"), (1.00, "100_percent")]

for fraction, label in configs:
    # Keep the same sampled dataset across repeats; repeated runs measure runtime variability.
    sample_seed = 2026 + int(fraction * 100)
    sampled = data.sample(withReplacement=False, fraction=fraction, seed=sample_seed).cache()
    sampled_count = sampled.count()

    for repeat in range(1, REPEATS + 1):
        train, test = sampled.randomSplit([0.8, 0.2], seed=42)
        train = train.cache()
        test = test.cache()
        train_count = train.count()
        test_count = test.count()

        if train_count == 0 or test_count == 0:
            train.unpersist()
            test.unpersist()
            continue

        total_start = time.perf_counter()

        global_mean = float(train.agg(F.avg("rating")).first()[0])
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

        feature_cols = [
            "user_avg_rating", "user_rating_count",
            "book_avg_rating", "book_rating_count"
        ]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        feature_start = time.perf_counter()
        train_features = assembler.transform(add_features(train)).cache()
        test_features = assembler.transform(add_features(test)).cache()
        train_features.count()
        test_features.count()
        feature_seconds = round(time.perf_counter() - feature_start, 2)

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
        training_seconds = round(time.perf_counter() - fit_start, 2)

        predict_start = time.perf_counter()
        predictions = model.transform(test_features).cache()
        prediction_count = predictions.count()
        prediction_seconds = round(time.perf_counter() - predict_start, 2)

        rmse = float(rmse_eval.evaluate(predictions))
        mae = float(mae_eval.evaluate(predictions))
        total_seconds = round(time.perf_counter() - total_start, 2)

        row = {
            "dataset_fraction": fraction,
            "dataset_label": label,
            "repeat": repeat,
            "input_rows": sampled_count,
            "train_rows": train_count,
            "test_rows": test_count,
            "prediction_rows": prediction_count,
            "coverage": prediction_count / test_count if test_count else 0.0,
            "feature_engineering_seconds": feature_seconds,
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "total_pipeline_seconds": total_seconds,
            "rmse": rmse,
            "mae": mae,
            "algorithm": "Spark MLlib RandomForestRegressor",
            "numTrees": 40,
            "maxDepth": 8,
            "sample_seed": sample_seed,
            "split_seed": 42
        }
        results.append(row)
        print(f"\n=== {label} | repeat {repeat}/{REPEATS} ===")
        print(json.dumps(row, indent=2))

        predictions.unpersist()
        train_features.unpersist()
        test_features.unpersist()
        train.unpersist()
        test.unpersist()

    sampled.unpersist()

# Calculate average and population standard deviation for each dataset size.
summaries = []
for fraction, label in configs:
    rows = [r for r in results if r["dataset_fraction"] == fraction]
    if not rows:
        continue
    summary = {
        "dataset_fraction": fraction,
        "dataset_label": label,
        "runs": len(rows),
        "input_rows": rows[0]["input_rows"],
    }
    for metric in [
        "feature_engineering_seconds", "training_seconds",
        "prediction_seconds", "total_pipeline_seconds", "rmse", "mae"
    ]:
        vals = [float(r[metric]) for r in rows]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        summary[f"{metric}_mean"] = round(mean, 4)
        summary[f"{metric}_stddev"] = round(variance ** 0.5, 4)
    summaries.append(summary)

payload = {
    "experiment": "Repeated single-machine data-volume scalability experiment",
    "input_parquet": INPUT,
    "available_explicit_interactions": total,
    "rating_filter": "Book-Rating 1 through 10; rating 0 excluded",
    "environment": {
        "spark_master": "local[*]",
        "spark_driver_memory": "2g",
        "algorithm": "Spark MLlib RandomForestRegressor",
        "numTrees": 40,
        "maxDepth": 8,
        "repeats_per_size": REPEATS,
        "feature_statistics_source": "training data only",
        "note": "Each size uses a fixed sample and fixed split across repeats; repeats measure runtime variability on the same data."
    },
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_results": results,
    "summary_mean_and_stddev": summaries
}

json_path = os.path.join(OUTPUT, "scalability_repeated_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

csv_path = os.path.join(OUTPUT, "scalability_repeated_runs.csv")
if results:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

summary_csv_path = os.path.join(OUTPUT, "scalability_summary.csv")
if summaries:
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

print("\n=== REPEATED SCALABILITY EXPERIMENT COMPLETE ===")
print(json.dumps({"summary_mean_and_stddev": summaries}, indent=2))
print(f"\nSaved JSON: {json_path}")
print(f"Saved runs CSV: {csv_path}")
print(f"Saved summary CSV: {summary_csv_path}")

data.unpersist()
spark.stop()
