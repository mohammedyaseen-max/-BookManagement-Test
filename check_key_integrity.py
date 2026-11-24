from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("BookCrossingKeyIntegrityCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

base = "data/processed"

for name, key in [
    ("books", "ISBN"),
    ("users", "User-ID")
]:
    df = spark.read.parquet(f"{base}/{name}")
    total = df.count()
    missing = df.filter(
        F.col(key).isNull() | (F.trim(F.col(key).cast("string")) == "")
    ).count()
    duplicate_groups = (
        df.groupBy(key).count()
        .filter(F.col("count") > 1)
        .count()
    )

    print(f"\n=== {name.upper()} ===")
    print("Total rows:", total)
    print(f"Missing {key}:", missing)
    print(f"Duplicate {key} groups:", duplicate_groups)

spark.stop()
