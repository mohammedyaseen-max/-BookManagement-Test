import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "raw"

def rows(name):
    with (DATA / name).open(encoding="latin-1", newline="") as f:
        yield from csv.DictReader(f)

print("=== BOOKS: YEAR AND KEY CHECKS ===")
years = Counter()
isbn_count = Counter()
for r in rows("Books.csv"):
    isbn_count[r["ISBN"].strip()] += 1
    try:
        years[int(r["Year-Of-Publication"].strip())] += 1
    except (ValueError, AttributeError):
        years["INVALID"] += 1

print("Duplicate ISBNs:", sum(n - 1 for n in isbn_count.values() if n > 1))
print("Year 0:", years[0])
print("Years before 0:", sum(n for y, n in years.items() if isinstance(y, int) and y < 0))
print("Years after 2026:", sum(n for y, n in years.items() if isinstance(y, int) and y > 2026))
print("Invalid years:", years["INVALID"])
print("Most common years:", years.most_common(12))

print("\n=== RATINGS: VALUE DISTRIBUTION ===")
rating_counts = Counter()
user_ids = set()
rating_isbns = set()
pairs = Counter()
for r in rows("Ratings.csv"):
    user = r["User-ID"].strip()
    isbn = r["ISBN"].strip()
    rating = r["Book-Rating"].strip()
    rating_counts[rating] += 1
    user_ids.add(user)
    rating_isbns.add(isbn)
    pairs[(user, isbn)] += 1

print("Rating counts:", dict(sorted(rating_counts.items())))
print("Rating 0 proportion:", round(rating_counts["0"] / sum(rating_counts.values()) * 100, 2), "%")
print("Repeated user-book pairs:", sum(n - 1 for n in pairs.values() if n > 1))
print("Unique users in ratings:", len(user_ids))
print("Unique ISBNs in ratings:", len(rating_isbns))

print("\n=== USERS: AGE CHECKS ===")
ages = Counter()
invalid_age = 0
for r in rows("Users.csv"):
    value = r["Age"].strip()
    if not value:
        ages["MISSING"] += 1
        continue
    try:
        age = float(value)
        if age < 0 or age > 120:
            ages["OUT_OF_RANGE"] += 1
        else:
            ages["VALID"] += 1
    except ValueError:
        invalid_age += 1

print("Age categories:", dict(ages))
print("Unparseable ages:", invalid_age)
