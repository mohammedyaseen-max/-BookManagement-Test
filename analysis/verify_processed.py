
from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("VerifyProcessedData")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "/project/data/processed"

results = []


def check(dataset, name, count, status, details=""):
    """Record and print one validation result."""
    results.append({
        "dataset": dataset,
        "check": name,
        "status": status,
        "count": int(count),
        "details": details
    })

    suffix = f" | {details}" if details else ""
    print(f"[{status}] {dataset} | {name}: {count}{suffix}")


def is_blank(column):
    return (
        F.col(column).isNull()
        | (F.trim(F.col(column).cast("string")) == "")
    )


try:
    datasets = {}

    # --------------------------------------------------
    # 1. Read and validate the four processed datasets
    # --------------------------------------------------
    for name in ["books", "users", "ratings", "ratings_enriched"]:
        print(f"\n=== {name.upper()} ===")

        path = f"{BASE}/{name}"
        df = spark.read.parquet(path)
        datasets[name] = df

        total = df.count()

        print("Rows:", total)
        print("Columns:", df.columns)

        check(
            name,
            "dataset_nonempty",
            total,
            "PASS" if total > 0 else "FAIL"
        )

        # ----------------------------------------------
        # 2. Books checks
        # ----------------------------------------------
        if name == "books":
            missing_isbn = df.filter(is_blank("ISBN")).count()

            duplicate_isbn_groups = (
                df.filter(~is_blank("ISBN"))
                .groupBy("ISBN")
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

            check(
                name,
                "missing_or_blank_ISBN",
                missing_isbn,
                "PASS" if missing_isbn == 0 else "FAIL"
            )

            check(
                name,
                "duplicate_ISBN_groups",
                duplicate_isbn_groups,
                "PASS" if duplicate_isbn_groups == 0 else "FAIL"
            )

        # ----------------------------------------------
        # 3. Users checks
        # ----------------------------------------------
        elif name == "users":
            missing_user_id = df.filter(is_blank("User-ID")).count()

            duplicate_user_id_groups = (
                df.filter(~is_blank("User-ID"))
                .groupBy("User-ID")
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

            # A source record without an ID is a warning.
            # It cannot be used in ID-based analysis.
            check(
                name,
                "missing_User_ID",
                missing_user_id,
                "WARNING" if missing_user_id > 0 else "PASS",
                (
                    "Source record without an ID; exclude from "
                    "ID-based analysis"
                    if missing_user_id > 0 else ""
                )
            )

            check(
                name,
                "duplicate_User_ID_groups",
                duplicate_user_id_groups,
                "PASS" if duplicate_user_id_groups == 0 else "FAIL"
            )

        # ----------------------------------------------
        # 4. Ratings and enriched ratings checks
        # ----------------------------------------------
        elif name in ["ratings", "ratings_enriched"]:
            missing_user_id = df.filter(is_blank("User-ID")).count()
            missing_isbn = df.filter(is_blank("ISBN")).count()

            invalid_or_null_rating = df.filter(
                F.col("Book-Rating").isNull()
                | ~F.col("Book-Rating").between(0, 10)
            ).count()

            duplicate_user_book_pairs = (
                df.groupBy("User-ID", "ISBN")
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

            check(
                name,
                "missing_User_ID",
                missing_user_id,
                "PASS" if missing_user_id == 0 else "FAIL"
            )

            check(
                name,
                "missing_ISBN",
                missing_isbn,
                "PASS" if missing_isbn == 0 else "FAIL"
            )

            check(
                name,
                "invalid_or_null_rating",
                invalid_or_null_rating,
                "PASS" if invalid_or_null_rating == 0 else "FAIL"
            )

            check(
                name,
                "duplicate_user_book_pairs",
                duplicate_user_book_pairs,
                "PASS" if duplicate_user_book_pairs == 0 else "FAIL"
            )

    # --------------------------------------------------
    # 5. Referential integrity checks
    # --------------------------------------------------
    print("\n=== REFERENTIAL INTEGRITY ===")

    books_keys = (
        datasets["books"]
        .filter(~is_blank("ISBN"))
        .select("ISBN")
        .distinct()
    )

    users_keys = (
        datasets["users"]
        .filter(~is_blank("User-ID"))
        .select("User-ID")
        .distinct()
    )

    enriched = datasets["ratings_enriched"]

    unmatched_book_groups = (
        enriched.select("ISBN")
        .distinct()
        .join(books_keys, on="ISBN", how="left_anti")
        .count()
    )

    unmatched_user_groups = (
        enriched.select("User-ID")
        .distinct()
        .join(users_keys, on="User-ID", how="left_anti")
        .count()
    )

    check(
        "ratings_enriched",
        "unmatched_book_ID_groups",
        unmatched_book_groups,
        "PASS" if unmatched_book_groups == 0 else "FAIL"
    )

    check(
        "ratings_enriched",
        "unmatched_user_ID_groups",
        unmatched_user_groups,
        "PASS" if unmatched_user_groups == 0 else "FAIL"
    )

    # --------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------
    print("\n=== FINAL VALIDATION SUMMARY ===")

    passed = sum(1 for result in results if result["status"] == "PASS")
    warnings = sum(
        1 for result in results if result["status"] == "WARNING"
    )
    failed = sum(1 for result in results if result["status"] == "FAIL")

    print("PASS:", passed)
    print("WARNING:", warnings)
    print("FAIL:", failed)
    print("TOTAL CHECKS:", len(results))

    if failed:
        print("\nRESULT: Validation failed. Investigate the FAIL checks.")
    elif warnings:
        print(
            "\nRESULT: Validation completed with warnings. "
            "Review and document the warnings."
        )
    else:
        print("\nRESULT: All validation checks passed.")

finally:
    spark.stop()
