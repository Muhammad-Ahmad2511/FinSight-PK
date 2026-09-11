import csv

files = ["dawn_business_2015_to_2019.csv", "dawn_business_2020_to_2022.csv", "dawn_business_2023_to_2026.csv"]

with open("dawn_business_2015_to_2026.csv", "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    for i, fname in enumerate(files):
        with open(fname, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            if i == 0:
                writer.writerow(header)
            writer.writerows(reader)

print("Done.")