import csv
import json
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType

spark = (
    SparkSession.builder
    .appName("BookCrossing-Data-Quality-Audit")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

BASE = "data/processed"
OUT = "analysis/results/data_quality_audit"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "books": os.path.join(BASE, "books"),
    "users": os.path.join(BASE, "users"),
    "ratings": os.path.join(BASE, "ratings"),
    "ratings_enriched": os.path.join(BASE, "ratings_enriched"),
}

def is_blank(col):
    return col.isNull() | (F.trim(col.cast("string")) == "")

summary_rows = []
column_rows = []
issue_rows = []

for name, path in DATASETS.items():
    if not os.path.exists(path):
        issue_rows.append({
            "dataset": name, "check": "dataset_path_exists",
            "status": "FAIL", "count": None,
            "details": f"Path not found: {path}"
        })
        continue

    df = spark.read.parquet(path)
    total = df.count()
    cols = df.columns

    # Row-level missingness: row is counted once if any column is null/blank.
    missing_any_expr = None
    for c in cols:
        expr = is_blank(F.col(c))
        missing_any_expr = expr if missing_any_expr is None else (missing_any_expr | expr)
    rows_with_missing = df.filter(missing_any_expr).count() if missing_any_expr is not None else 0

    # Exact duplicate rows (all columns equal).
    duplicate_rows = total - df.dropDuplicates().count()

    summary_rows.append({
        "dataset": name,
        "path": path,
        "rows": total,
        "columns": len(cols),
        "rows_with_any_missing_or_blank": rows_with_missing,
        "exact_duplicate_rows": duplicate_rows
    })

    for c in cols:
        null_count = df.filter(F.col(c).isNull()).count()
        blank_count = df.filter(F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) == "")).count()
        column_rows.append({
            "dataset": name,
            "column": c,
            "data_type": dict(df.dtypes).get(c, "unknown"),
            "null_count": null_count,
            "blank_count": blank_count,
            "null_or_blank_pct": round((null_count + blank_count) / total * 100, 4) if total else 0
        })

    issue_rows.append({
        "dataset": name, "check": "exact_duplicate_rows",
        "status": "PASS" if duplicate_rows == 0 else "REVIEW",
        "count": duplicate_rows,
        "details": "Exact duplicate rows across all columns"
    })

    if name in ("ratings", "ratings_enriched"):
        if "Book-Rating" in cols:
            invalid_rating = df.filter(
                F.col("Book-Rating").isNotNull()
                & ~F.col("Book-Rating").cast("int").between(0, 10)
            ).count()
            issue_rows.append({
                "dataset": name, "check": "rating_outside_0_to_10",
                "status": "PASS" if invalid_rating == 0 else "FAIL",
                "count": invalid_rating,
                "details": "Rating values outside 0..10"
            })
            zero_rating = df.filter(F.col("Book-Rating").cast("int") == 0).count()
            explicit_rating = df.filter(F.col("Book-Rating").cast("int").between(1, 10)).count()
            issue_rows.append({
                "dataset": name, "check": "zero_rating_rows",
                "status": "INFO", "count": zero_rating,
                "details": "Book-Crossing rating 0 treated separately from explicit ratings"
            })
            issue_rows.append({
                "dataset": name, "check": "explicit_rating_rows_1_to_10",
                "status": "INFO", "count": explicit_rating,
                "details": "Explicit ratings available for rating prediction"
            })

    if name == "books" and "ISBN" in cols:
        bad_isbn = df.filter(is_blank(F.col("ISBN"))).count()
        issue_rows.append({
            "dataset": name, "check": "missing_or_blank_ISBN",
            "status": "REVIEW" if bad_isbn else "PASS",
            "count": bad_isbn,
            "details": "Missing/blank book identifiers"
        })

    if name == "users" and "User-ID" in cols:
        bad_user_id = df.filter(is_blank(F.col("User-ID"))).count()
        issue_rows.append({
            "dataset": name, "check": "missing_or_blank_User-ID",
            "status": "REVIEW" if bad_user_id else "PASS",
            "count": bad_user_id,
            "details": "Missing/blank user identifiers"
        })

# Write reports using the standard library so pandas is not needed in the container.
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(os.path.join(OUT, "dataset_summary.csv"), summary_rows)
write_csv(os.path.join(OUT, "column_missingness.csv"), column_rows)
write_csv(os.path.join(OUT, "quality_checks.csv"), issue_rows)

report = {
    "audit": "Book-Crossing processed dataset quality audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "datasets_checked": list(DATASETS.keys()),
    "summary": summary_rows,
    "column_missingness": column_rows,
    "checks": issue_rows,
    "interpretation_notes": [
        "Exact duplicates compare all columns; repeated user-book ratings are not automatically duplicates unless all columns match.",
        "Rating 0 is reported separately and should not be treated as an explicit rating in rating-prediction experiments.",
        "REVIEW means inspect the count and decide whether it is expected; it is not automatically a pipeline failure."
    ]
}
with open(os.path.join(OUT, "data_quality_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n=== DATA QUALITY AUDIT COMPLETE ===")
print(json.dumps({"summary": summary_rows, "checks": issue_rows}, indent=2, ensure_ascii=False))
print(f"\nSaved reports under: {OUT}")

spark.stop()
