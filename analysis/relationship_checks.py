import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

def load_keys(filename, column):
    result = set()
    with (DATA / filename).open(encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f):
            result.add(row[column].strip())
    return result

book_isbns = load_keys("Books.csv", "ISBN")
user_ids = load_keys("Users.csv", "User-ID")

total = 0
matched_books = 0
matched_users = 0
matched_both = 0

with (DATA / "Ratings.csv").open(encoding="latin-1", newline="") as f:
    for row in csv.DictReader(f):
        total += 1
        book_ok = row["ISBN"].strip() in book_isbns
        user_ok = row["User-ID"].strip() in user_ids

        matched_books += book_ok
        matched_users += user_ok
        matched_both += book_ok and user_ok

print("Total ratings:", total)
print("Ratings with known books:", matched_books,
      f"({matched_books / total * 100:.2f}%)")
print("Ratings with known users:", matched_users,
      f"({matched_users / total * 100:.2f}%)")
print("Ratings with both known book and user:", matched_both,
      f"({matched_both / total * 100:.2f}%)")
