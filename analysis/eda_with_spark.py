from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spark = (
    SparkSession.builder
    .appName("BookCrossingEDA")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

base = str(ROOT / "data" / "processed")
out = str(ROOT / "analysis" / "results")

books = spark.read.parquet(f"{base}/books")
users = spark.read.parquet(f"{base}/users")
ratings = spark.read.parquet(f"{base}/ratings")

# Exclude records with missing identifiers from analytical joins.
books = books.filter(
    F.col("ISBN").isNotNull() & (F.trim("ISBN") != "")
)
users = users.filter(
    F.col("User-ID").isNotNull() & (F.trim("User-ID") != "")
)
ratings = ratings.filter(
    F.col("User-ID").isNotNull()
    & (F.trim("User-ID") != "")
    & F.col("ISBN").isNotNull()
    & (F.trim("ISBN") != "")
)

# Explicit ratings only: 1-10. Rating 0 remains in the source data.
explicit = ratings.filter(F.col("Book-Rating").between(1, 10))

joined = (
    explicit
    .join(books.select("ISBN", "Book-Title"), "ISBN", "inner")
    .join(users.select("User-ID"), "User-ID", "inner")
)

print("\n=== DATASET COUNTS ===")
print("Books:", books.count())
print("Users with valid IDs:", users.count())
print("All valid-identifier ratings:", ratings.count())
print("Explicit ratings (1-10):", explicit.count())
print("Explicit ratings joined to known books and users:", joined.count())

print("\n=== QUESTION 1: MOST RATED BOOKS ===")
most_rated = (
    joined.groupBy("ISBN", "Book-Title")
    .agg(
        F.count("*").alias("rating_count"),
        F.round(F.avg("Book-Rating"), 3).alias("average_rating")
    )
    .orderBy(F.desc("rating_count"), F.asc("Book-Title"))
)

most_rated.show(10, truncate=60)

print("\n=== QUESTION 2: HIGHEST RATED BOOKS ===")
best_rated = (
    most_rated.filter(F.col("rating_count") >= 50)
    .orderBy(F.desc("average_rating"), F.desc("rating_count"))
)

best_rated.show(10, truncate=60)

print("\n=== QUESTION 3: EXPLICIT RATING DISTRIBUTION ===")
distribution = (
    explicit.groupBy("Book-Rating")
    .count()
    .orderBy("Book-Rating")
)
distribution.show()

print("\n=== QUESTION 4: MOST ACTIVE USERS ===")
active_users = (
    explicit.groupBy("User-ID")
    .count()
    .withColumnRenamed("count", "explicit_rating_count")
    .orderBy(F.desc("explicit_rating_count"))
)
active_users.show(10)

# Save results as CSV folders, each containing part files.
most_rated.limit(100).coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(f"{out}/most_rated_books")

best_rated.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(f"{out}/highest_rated_books")

distribution.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(f"{out}/rating_distribution")

active_users.limit(100).coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(f"{out}/most_active_users")

print("\nSaved analytical results to:", out)
spark.stop()
