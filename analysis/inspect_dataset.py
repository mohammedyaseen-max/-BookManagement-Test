import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "raw"
FILES = ["Books.csv", "Ratings.csv", "Users.csv"]

for filename in FILES:
    path = DATA / filename
    print("\n" + "=" * 65)
    print(f"FILE: {filename} | SIZE: {path.stat().st_size / 1048576:.2f} MB")

    with path.open("r", encoding="latin-1", newline="") as f:
        sample = f.read(65536)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader, [])
        counts = Counter()
        seen = set()
        total = duplicates = malformed = 0
        examples = []

        print("DELIMITER:", repr(delimiter))
        print("COLUMNS:", headers)

        for row in reader:
            total += 1

            if len(row) != len(headers):
                malformed += 1

            for i, name in enumerate(headers):
                value = row[i].strip() if i < len(row) and row[i] else ""
                if not value or value.upper() in {"NULL", "NA", "N/A"}:
                    counts[name] += 1

            key = tuple(row)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

            if len(examples) < 2:
                examples.append(row)

    print(f"ROWS: {total:,}")
    print(f"EXACT DUPLICATE ROWS: {duplicates:,}")
    print(f"ROWS WITH UNEXPECTED COLUMN COUNT: {malformed:,}")
    print("EMPTY / NULL VALUES:")
    for name in headers:
        print(f"  {name}: {counts[name]:,}")
    print("SAMPLE ROWS:")
    for row in examples:
        print(row)
