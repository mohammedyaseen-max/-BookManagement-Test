from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("InspectMissingUserID")
    .getOrCreate()
)

users = spark.read.parquet("data/processed/users")

missing_users = users.filter(
    F.col("User-ID").isNull()
    | (F.trim(F.col("User-ID").cast("string")) == "")
)

print("\n=== Missing User-ID records ===")
missing_users.show(10, truncate=False)

print("\n=== Number of affected records ===")
print(missing_users.count())

spark.stop()
