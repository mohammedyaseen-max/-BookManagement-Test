from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("BookCrossingDuplicateRatingCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/processed/ratings")

pairs = (
    df.groupBy("User-ID", "ISBN")
    .agg(
        F.count("*").alias("records"),
        F.countDistinct("Book-Rating").alias("distinct_ratings")
    )
)

print("Total rating rows:", df.count())
print("Repeated user-book pairs:", pairs.filter(F.col("records") > 1).count())
print(
    "Pairs with conflicting ratings:",
    pairs.filter(F.col("distinct_ratings") > 1).count()
)

pairs.filter(F.col("records") > 1).orderBy(
    F.desc("records")
).show(20, truncate=False)

spark.stop()
