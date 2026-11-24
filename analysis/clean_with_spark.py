ROOT = Path(__file__).resolve().parents[1]
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("BookCrossingDataQuality")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

base = str(ROOT / "data" / "raw")
out = str(ROOT / "data" / "processed")

print("\n=== 1. READ DATA ===")

books = (
    spark.read
    .option("header", True)
    .option("encoding", "ISO-8859-1")
    .option("inferSchema", False)
    .csv(f"{base}/Books.csv")
)

ratings = (
    spark.read
    .option("header", True)
    .option("encoding", "ISO-8859-1")
    .option("inferSchema", False)
    .csv(f"{base}/Ratings.csv")
)

users = (
    spark.read
    .option("header", True)
    .option("encoding", "ISO-8859-1")
    .option("inferSchema", False)
    .csv(f"{base}/Users.csv")
)

print("Books:", books.count(), "rows")
print("Ratings:", ratings.count(), "rows")
print("Users:", users.count(), "rows")

print("\n=== 2. CLEAN BOOKS ===")

books = (
    books
    .withColumn("ISBN", F.trim(F.col("ISBN")))
    .withColumn("Book-Title", F.trim(F.col("Book-Title")))
    .withColumn("Book-Author", F.trim(F.col("Book-Author")))
    .withColumn(
        "Year-Of-Publication",
        F.col("Year-Of-Publication").cast("int")
    )
)

books = books.withColumn(
    "Year-Of-Publication",
    F.when(
        (F.col("Year-Of-Publication") >= 1450) &
        (F.col("Year-Of-Publication") <= 2026),
        F.col("Year-Of-Publication")
    ).otherwise(None)
)

# Keep one record per ISBN for joining.
# Prefer records with a title and a valid publication year.
books = (
    books
    .withColumn(
        "_quality",
        F.when(F.col("Book-Title").isNotNull() &
               (F.col("Book-Title") != ""), 1).otherwise(0)
        + F.when(F.col("Year-Of-Publication").isNotNull(), 1)
          .otherwise(0)
    )
    .orderBy(F.col("_quality").desc())
    .dropDuplicates(["ISBN"])
    .drop("_quality")
)

print("Books after ISBN deduplication:", books.count())

print("\n=== 3. CLEAN USERS ===")

users = (
    users
    .withColumn("User-ID", F.trim(F.col("User-ID")))
    .withColumn("Age", F.col("Age").cast("double"))
    .withColumn(
        "Age",
        F.when(
            (F.col("Age") >= 1) & (F.col("Age") <= 120),
            F.col("Age")
        ).otherwise(None)
    )
    .dropDuplicates(["User-ID"])
)

print("Users after ID deduplication:", users.count())

print("\n=== 4. CLEAN RATINGS ===")

ratings = (
    ratings
    .withColumn("User-ID", F.trim(F.col("User-ID")))
    .withColumn("ISBN", F.trim(F.col("ISBN")))
    .withColumn("Book-Rating", F.col("Book-Rating").cast("int"))
    .filter(F.col("User-ID").isNotNull() & (F.col("User-ID") != ""))
    .filter(F.col("ISBN").isNotNull() & (F.col("ISBN") != ""))
    .filter(F.col("Book-Rating").between(0, 10))
    .dropDuplicates(["User-ID", "ISBN"])
)

print("Ratings after basic cleaning:", ratings.count())

print("\n=== 5. JOIN AND REPORT ===")

rating_count = ratings.count()

ratings_with_books = ratings.join(
    books.select("ISBN", "Book-Title", "Book-Author",
                 "Year-Of-Publication", "Publisher"),
    on="ISBN",
    how="inner"
)

ratings_with_users = ratings_with_books.join(
    users.select("User-ID", "Age", "Location"),
    on="User-ID",
    how="inner"
)

print("Ratings before book join:", rating_count)
print("Ratings with known books:", ratings_with_books.count())
print("Ratings with known books and users:", ratings_with_users.count())

print("\n=== 6. WRITE PROCESSED DATA ===")

books.write.mode("overwrite").parquet(f"{out}/books")
users.write.mode("overwrite").parquet(f"{out}/users")
ratings.write.mode("overwrite").parquet(f"{out}/ratings")
ratings_with_users.write.mode("overwrite").parquet(f"{out}/ratings_enriched")

print("Saved Parquet datasets to:", out)

print("\n=== 7. RATING DISTRIBUTION ===")

ratings.groupBy("Book-Rating").count().orderBy("Book-Rating").show()

spark.stop()
