from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import json
import os
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("BookCrossing-ALS-Tuning")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

INPUT = str(ROOT / "data" / "processed" / "ratings_enriched")
OUTPUT = str(ROOT / "analysis" / "results" / "als_tuning")
os.makedirs(OUTPUT, exist_ok=True)

ratings = spark.read.parquet(INPUT)

# Keep explicit ratings only and validate numeric user IDs.
ratings = (
    ratings
    .select(
        F.col("User-ID").cast("int").alias("user_id"),
        F.trim(F.col("ISBN")).alias("isbn"),
        F.col("Book-Rating").cast("float").alias("rating")
    )
    .filter(
        F.col("user_id").isNotNull()
        & F.col("isbn").isNotNull()
        & (F.col("isbn") != "")
        & F.col("rating").between(1, 10)
    )
)

item_indexer = StringIndexer(
    inputCol="isbn", outputCol="item_id", handleInvalid="skip"
)
indexed = item_indexer.fit(ratings).transform(ratings)
indexed = indexed.select(
    "user_id",
    F.col("item_id").cast("int").alias("item_id"),
    "rating"
).cache()

train, test = indexed.randomSplit([0.8, 0.2], seed=42)
train = train.cache()
test = test.cache()

train.count()
test.count()

evaluator_rmse = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="rating", predictionCol="prediction", metricName="mae"
)

# Evaluate all models on the same test rows where every model predicts.
test_users = train.select("user_id").distinct()
test_items = train.select("item_id").distinct()
common_test = (
    test.join(test_users, "user_id", "inner")
        .join(test_items, "item_id", "inner")
        .cache()
)
common_count = common_test.count()

global_mean = float(train.agg(F.avg("rating")).first()[0])
baseline = common_test.withColumn("prediction", F.lit(global_mean))
baseline_rmse = float(evaluator_rmse.evaluate(baseline))
baseline_mae = float(evaluator_mae.evaluate(baseline))

configs = [
    {"rank": 10, "regParam": 0.1, "maxIter": 10},
    {"rank": 20, "regParam": 0.1, "maxIter": 10},
    {"rank": 10, "regParam": 0.05, "maxIter": 15},
]

results = []
for cfg in configs:
    print(f"\nTraining ALS with {cfg}", flush=True)
    start = time.time()

    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        rank=cfg["rank"],
        regParam=cfg["regParam"],
        maxIter=cfg["maxIter"],
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42
    )

    model = als.fit(train)
    predictions = model.transform(common_test).cache()
    n = predictions.count()

    if n:
        rmse = float(evaluator_rmse.evaluate(predictions))
        mae = float(evaluator_mae.evaluate(predictions))
    else:
        rmse, mae = None, None

    result = {
        **cfg,
        "common_test_rows": common_count,
        "als_evaluable_rows": n,
        "coverage_on_common_test": n / common_count if common_count else 0,
        "rmse": rmse,
        "mae": mae,
        "training_seconds": round(time.time() - start, 2),
        "baseline_rmse_same_test": baseline_rmse,
        "baseline_mae_same_test": baseline_mae
    }
    results.append(result)

    print(json.dumps(result, indent=2), flush=True)
    predictions.unpersist()
    model = None

with open(os.path.join(OUTPUT, "tuning_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n=== BASELINE ON COMMON TEST ROWS ===")
print(json.dumps({
    "common_test_rows": common_count,
    "global_mean": global_mean,
    "rmse": baseline_rmse,
    "mae": baseline_mae
}, indent=2))

print(f"\nResults saved to {OUTPUT}/tuning_results.json")

common_test.unpersist()
train.unpersist()
test.unpersist()
indexed.unpersist()
spark.stop()
