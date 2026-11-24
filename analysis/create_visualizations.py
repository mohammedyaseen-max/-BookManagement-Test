
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.dpi": 250,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "font.size": 10,
})


def read_csv_folder(folder):
    files = sorted((RESULTS / folder).glob("part-*.csv"))
    if not files:
        raise FileNotFoundError(f"No part CSV found in {folder}")

    with files[0].open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / name}")


# 1. Rating distribution
rows = read_csv_folder("rating_distribution")
rows = sorted(rows, key=lambda r: int(r["Book-Rating"]))

ratings = [int(r["Book-Rating"]) for r in rows]
counts = [int(r["count"]) for r in rows]

plt.figure()
plt.bar(ratings, counts)
plt.xticks(range(1, 11))
plt.xlabel("Book Rating")
plt.ylabel("Number of Ratings")
plt.title("Distribution of Explicit Book Ratings (1-10)")
plt.grid(axis="y", alpha=0.25)
save("01_rating_distribution.png")


# 2. Top 10 most-rated books
rows = read_csv_folder("most_rated_books")
rows = sorted(
    rows,
    key=lambda r: int(r["rating_count"]),
    reverse=True
)[:10]
rows.reverse()

labels = [
    (r["Book-Title"][:38] + "...")
    if len(r["Book-Title"]) > 38
    else r["Book-Title"]
    for r in rows
]
values = [int(r["rating_count"]) for r in rows]

plt.figure(figsize=(11, 7))
plt.barh(labels, values)
plt.xlabel("Number of Ratings")
plt.ylabel("Book")
plt.title("Top 10 Most-Rated Books")
plt.grid(axis="x", alpha=0.25)
save("02_top_10_most_rated_books.png")


# 3. Popularity versus average rating
rows = read_csv_folder("popularity_vs_average_rating")
points = []

for r in rows:
    try:
        n = int(r["rating_count"])
        avg = float(r["average_rating"])
        if n > 0:
            points.append((n, avg))
    except (ValueError, KeyError):
        pass

if points:
    random.seed(42)
    if len(points) > 5000:
        points = random.sample(points, 5000)

    x, y = zip(*points)

    plt.figure()
    plt.scatter(x, y, alpha=0.55, s=18)
    plt.xscale("log")
    plt.xlabel("Number of Ratings (log scale)")
    plt.ylabel("Average Book Rating")
    plt.title("Book Popularity vs. Average Rating")
    plt.grid(alpha=0.25)
    save("03_popularity_vs_average_rating.png")


# 4. User activity groups, in logical order
rows = read_csv_folder("user_activity_distribution")

activity_order = {
    "1 rating": 1,
    "2-5 ratings": 2,
    "6-20 ratings": 3,
    "21-100 ratings": 4,
    "More than 100 ratings": 5,
}

rows.sort(
    key=lambda r: activity_order.get(r["activity_group"], 999)
)

labels = [r["activity_group"] for r in rows]
values = [int(r["user_count"]) for r in rows]

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.xlabel("Number of Ratings per User")
plt.ylabel("Number of Users")
plt.title("User Activity Distribution")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)
save("04_user_activity_distribution.png")


# 5. Books by publication year: focused display period
rows = read_csv_folder("books_by_publication_year")
year_counts = []

for r in rows:
    try:
        year = int(float(r["Year-Of-Publication"]))
        count = int(r["book_count"])

        if 1450 <= year <= 2026:
            year_counts.append((year, count))

    except (ValueError, KeyError):
        pass

year_counts.sort()

# Focused display only; source data remains unchanged.
main_period = [
    (year, count)
    for year, count in year_counts
    if 1950 <= year <= 2006
]

plt.figure(figsize=(12, 6))
plt.plot(
    [r[0] for r in main_period],
    [r[1] for r in main_period],
    marker=".",
    markersize=3,
    linewidth=1.3
)
plt.xlabel("Publication Year")
plt.ylabel("Number of Books")
plt.title("Number of Books by Publication Year (1950-2006)")
plt.xlim(1950, 2006)
plt.grid(alpha=0.25)
save("05_books_by_publication_year.png")


# 6. Model comparison on all test rows
with (
    RESULTS / "fair_model_comparison" /
    "comparison_metrics.json"
).open("r", encoding="utf-8") as f:
    metrics = json.load(f)

full = metrics["all_test_rows"]

model_names = [
    "Global Mean Baseline",
    "Random Forest",
    "ALS"
]
model_keys = [
    "global_mean_baseline",
    "random_forest",
    "als"
]

rmse = [full[k]["rmse"] for k in model_keys]
mae = [full[k]["mae"] for k in model_keys]

coverage = [
    full[k].get("coverage", 1.0) * 100
    for k in model_keys
]

plt.figure(figsize=(10, 6))
positions = list(range(len(model_names)))
width = 0.36

plt.bar(
    [p - width / 2 for p in positions],
    rmse,
    width,
    label="RMSE"
)
plt.bar(
    [p + width / 2 for p in positions],
    mae,
    width,
    label="MAE"
)

plt.xticks(positions, model_names)
plt.ylabel("Prediction Error (lower is better)")
plt.title("Rating Prediction Model Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.25)

for i, cov in enumerate(coverage):
    plt.text(
        i,
        max(rmse[i], mae[i]) + 0.05,
        f"Coverage: {cov:.1f}%",
        ha="center",
        fontsize=9
    )

save("06_model_comparison.png")


# 7. Scalability: runtime and prediction quality
with (
    RESULTS / "scalability_experiment_repeated" /
    "scalability_summary.csv"
).open("r", encoding="utf-8-sig", newline="") as f:
    scale_rows = list(csv.DictReader(f))

scale_rows.sort(key=lambda r: float(r["dataset_fraction"]))

fractions = [
    float(r["dataset_fraction"]) * 100
    for r in scale_rows
]
runtime = [
    float(r["total_pipeline_seconds_mean"])
    for r in scale_rows
]
runtime_std = [
    float(r["total_pipeline_seconds_stddev"])
    for r in scale_rows
]
scale_rmse = [float(r["rmse_mean"]) for r in scale_rows]
scale_mae = [float(r["mae_mean"]) for r in scale_rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].errorbar(
    fractions,
    runtime,
    yerr=runtime_std,
    marker="o",
    capsize=4
)
axes[0].set_xlabel("Dataset Size (%)")
axes[0].set_ylabel("Mean Pipeline Runtime (seconds)")
axes[0].set_title("Runtime vs. Dataset Size")
axes[0].grid(alpha=0.25)

axes[1].plot(
    fractions, scale_rmse,
    marker="o", label="RMSE"
)
axes[1].plot(
    fractions, scale_mae,
    marker="s", label="MAE"
)
axes[1].set_xlabel("Dataset Size (%)")
axes[1].set_ylabel("Prediction Error")
axes[1].set_title("Prediction Error vs. Dataset Size")
axes[1].legend()
axes[1].grid(alpha=0.25)

save("07_scalability_analysis.png")


# 8. Fair model comparison on the same 48,016 test rows
common = metrics["same_test_rows_comparison"]

common_rmse = [
    common[k]["rmse"]
    for k in model_keys
]
common_mae = [
    common[k]["mae"]
    for k in model_keys
]

plt.figure(figsize=(10, 6))

positions = list(range(len(model_names)))
width = 0.36

plt.bar(
    [p - width / 2 for p in positions],
    common_rmse,
    width,
    label="RMSE"
)
plt.bar(
    [p + width / 2 for p in positions],
    common_mae,
    width,
    label="MAE"
)

plt.xticks(positions, model_names)
plt.ylabel("Prediction Error (lower is better)")
plt.title("Fair Model Comparison on Common Test Rows (n=48,016)")
plt.legend()
plt.grid(axis="y", alpha=0.25)

save("08_fair_model_comparison_common_rows.png")


print(f"\nFinished. Figures saved in: {OUT}")
