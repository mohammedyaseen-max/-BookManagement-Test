from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("InspectMissingUserID")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

users = spark.read.parquet("data/processed/users")

users.filter(
    F.col("User-ID").isNull()
    | (F.trim(F.col("User-ID").cast("string")) == "")
).show(10, truncate=False)

spark.stop()
