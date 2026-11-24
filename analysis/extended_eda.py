from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spark = (
    SparkSession.builder
    .appName("BookCrossingExtendedEDA")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

base = str(ROOT / "data" / "processed")
out = str(ROOT / "analysis" / "results")

books = spark.read.parquet(f"{base}/books")
users = spark.read.parquet(f"{base}/users")
ratings = spark.read.parquet(f"{base}/ratings")

# Valid identifiers and explicit ratings only
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

explicit = ratings.filter(F.col("Book-Rating").between(1, 10))

# Join only the columns needed for these analyses
joined = (
    explicit
    .join(
        books.select(
            "ISBN", "Book-Title", "Book-Author",
            "Year-Of-Publication"
        ),
        "ISBN",
        "inner"
    )
    .join(users.select("User-ID"), "User-ID", "inner")
)

def save_csv(df, name):
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"{out}/{name}")
    )
    print(f"Saved: {out}/{name}")

print("\n=== ANALYSIS 5: BOOK PUBLICATION YEARS ===")
years = (
    books
    .filter(F.col("Year-Of-Publication").isNotNull())
    .groupBy("Year-Of-Publication")
    .agg(F.count("*").alias("book_count"))
    .orderBy(F.desc("book_count"))
)
years.show(15)
save_csv(years, "books_by_publication_year")

print("\n=== ANALYSIS 6: AUTHORS WITH MOST EXPLICIT RATINGS ===")
authors = (
    joined
    .filter(
        F.col("Book-Author").isNotNull()
        & (F.trim("Book-Author") != "")
    )
    .groupBy("Book-Author")
    .agg(
        F.count("*").alias("rating_count"),
        F.round(F.avg("Book-Rating"), 3).alias("average_rating"),
        F.countDistinct("ISBN").alias("distinct_books")
    )
    .filter(F.col("rating_count") >= 50)
    .orderBy(F.desc("rating_count"))
)
authors.show(15, truncate=50)
save_csv(authors, "authors_by_rating_count")

print("\n=== ANALYSIS 7: RATINGS BY PUBLICATION YEAR ===")
ratings_by_year = (
    joined
    .filter(F.col("Year-Of-Publication").isNotNull())
    .groupBy("Year-Of-Publication")
    .agg(
        F.count("*").alias("rating_count"),
        F.round(F.avg("Book-Rating"), 3).alias("average_rating"),
        F.countDistinct("ISBN").alias("distinct_books")
    )
    .filter(F.col("rating_count") >= 100)
    .orderBy("Year-Of-Publication")
)
ratings_by_year.show(20)
save_csv(ratings_by_year, "ratings_by_publication_year")

print("\n=== ANALYSIS 8: USER ACTIVITY DISTRIBUTION ===")
activity = (
    explicit
    .groupBy("User-ID")
    .agg(F.count("*").alias("rating_count"))
)

activity_groups = (
    activity
    .withColumn(
        "activity_group",
        F.when(F.col("rating_count") == 1, "1 rating")
        .when(F.col("rating_count").between(2, 5), "2-5 ratings")
        .when(F.col("rating_count").between(6, 20), "6-20 ratings")
        .when(F.col("rating_count").between(21, 100), "21-100 ratings")
        .otherwise("More than 100 ratings")
    )
    .groupBy("activity_group")
    .agg(
        F.count("*").alias("user_count"),
        F.round(F.avg("rating_count"), 2).alias("average_ratings_per_user")
    )
    .orderBy("activity_group")
)
activity_groups.show(truncate=False)
save_csv(activity_groups, "user_activity_distribution")

print("\n=== ANALYSIS 9: POPULARITY VS AVERAGE RATING ===")
book_metrics = (
    joined
    .groupBy("ISBN", "Book-Title")
    .agg(
        F.count("*").alias("rating_count"),
        F.round(F.avg("Book-Rating"), 3).alias("average_rating")
    )
)

# Save all books with at least 20 explicit ratings.
# This lets us compare popularity and average rating later.
popularity_vs_rating = (
    book_metrics
    .filter(F.col("rating_count") >= 20)
    .orderBy(F.desc("rating_count"))
)
popularity_vs_rating.show(15, truncate=50)
save_csv(popularity_vs_rating, "popularity_vs_average_rating")

print("\nAll extended EDA analyses completed.")
spark.stop()
