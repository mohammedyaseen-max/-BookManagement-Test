"""Inspect processed user records with missing or blank User-ID values."""
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "users"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Processed users dataset not found: {INPUT}. "
            "Run analysis/clean_with_spark.py first."
        )

    spark = (
        SparkSession.builder
        .appName("InspectMissingUserID")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        users = spark.read.parquet(str(INPUT))
        missing_users = users.filter(
            F.col("User-ID").isNull()
            | (F.trim(F.col("User-ID").cast("string")) == "")
        )
        print("\n=== Missing User-ID records ===")
        missing_users.show(10, truncate=False)
        print("\n=== Number of affected records ===")
        print(missing_users.count())
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
