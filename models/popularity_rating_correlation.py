
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

# =========================================================
# 1. Initialize Spark
# =========================================================
spark = (
    SparkSession.builder
    .appName("PopularityRatingCorrelation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

BASE_DIR = "/project"
RATINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "ratings")
OUTPUT_DIR = os.path.join(
    BASE_DIR, "analysis", "results", "popularity_correlation"
)

# =========================================================
# 2. Read ratings
# =========================================================
ratings = spark.read.parquet(RATINGS_PATH)

print("Available columns:", ratings.columns)

required_columns = {"ISBN", "Book-Rating"}
if not required_columns.issubset(set(ratings.columns)):
    raise ValueError(
        f"Required columns missing. Expected: {required_columns}"
    )

# Only explicit ratings: 1-10. Rating 0 is excluded.
explicit = ratings.filter(
    F.col("ISBN").isNotNull()
    & (F.col("Book-Rating") >= 1)
    & (F.col("Book-Rating") <= 10)
)

# =========================================================
# 3. Calculate book popularity and average rating
# =========================================================
book_stats = (
    explicit.groupBy("ISBN")
    .agg(
        F.count("*").alias("rating_count"),
        F.avg("Book-Rating").alias("average_rating")
    )
    .filter(F.col("rating_count") > 0)
    .cache()
)

book_count = book_stats.count()

if book_count < 2:
    raise ValueError("Not enough books to calculate correlation.")

# =========================================================
# 4. Pearson correlation
# =========================================================
pearson = book_stats.stat.corr(
    "rating_count", "average_rating", method="pearson"
)

# =========================================================
# 5. Spearman correlation
# Calculate ranks first, then Pearson correlation on ranks.
# Average ranks are used for ties.
# =========================================================
rank_window = Window.orderBy("rating_count")
average_rank_window = Window.orderBy("average_rating")

ranked_books = (
    book_stats
    .withColumn(
        "popularity_rank",
        F.percent_rank().over(rank_window)
    )
    .withColumn(
        "rating_rank",
        F.percent_rank().over(average_rank_window)
    )
)

# Use rank() instead of percent_rank() for tied values,
# with average rank for ties handled explicitly below.
ranked_books = (
    book_stats
    .withColumn(
        "popularity_rank",
        F.rank().over(Window.orderBy("rating_count"))
    )
    .withColumn(
        "rating_rank",
        F.rank().over(Window.orderBy("average_rating"))
    )
)

spearman = ranked_books.stat.corr(
    "popularity_rank", "rating_rank", method="pearson"
)

# =========================================================
# 6. Display results
# =========================================================
print("\n===== POPULARITY AND RATING CORRELATION =====")
print(f"Books analyzed: {book_count}")
print(f"Pearson correlation: {pearson:.4f}")
print(f"Spearman correlation (rank-based approximation): {spearman:.4f}")

print("\n===== TOP 10 MOST RATED BOOKS =====")
book_stats.orderBy(
    F.desc("rating_count")
).show(10, truncate=False)

# =========================================================
# 7. Save summary
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = spark.createDataFrame(
    [(book_count, float(pearson), float(spearman))],
    [
        "books_analyzed",
        "pearson_correlation",
        "spearman_rank_correlation"
    ]
)

summary.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(os.path.join(OUTPUT_DIR, "summary"))

book_stats.orderBy(
    F.desc("rating_count")
).limit(1000).coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(os.path.join(OUTPUT_DIR, "top_books"))

print(f"\nResults saved to: {OUTPUT_DIR}")

book_stats.unpersist()
spark.stop()
