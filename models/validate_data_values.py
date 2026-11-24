from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("ValidateBookCrossingValues")
    .getOrCreate()
)

books = spark.read.parquet(str(ROOT / "data" / "processed" / "books"))
users = spark.read.parquet(str(ROOT / "data" / "processed" / "users"))

print("\n=== BOOK PUBLICATION YEAR CHECK ===")

year = F.col("Year-Of-Publication").cast("int")

books.select(
    F.count("*").alias("total_rows"),
    F.sum(F.when(year.isNull(), 1).otherwise(0))
     .alias("missing_or_non_numeric"),
    F.sum(F.when((year < 1450) | (year > 2026), 1).otherwise(0))
     .alias("outside_1450_2026"),
    F.sum(F.when(year == 0, 1).otherwise(0))
     .alias("year_zero")
).show(truncate=False)

print("\n=== MOST COMMON PUBLICATION YEARS ===")

books.groupBy("Year-Of-Publication") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(20, truncate=False)

print("\n=== USER AGE CHECK ===")

age = F.col("Age").cast("double")

users.select(
    F.count("*").alias("total_rows"),
    F.sum(F.when(age.isNull(), 1).otherwise(0))
     .alias("missing_or_non_numeric"),
    F.sum(F.when((age < 5) | (age > 100), 1).otherwise(0))
     .alias("outside_5_100"),
    F.min(age).alias("minimum_age"),
    F.max(age).alias("maximum_age")
).show(truncate=False)

print("\n=== MOST COMMON AGES ===")

users.groupBy("Age") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(20, truncate=False)

spark.stop()
