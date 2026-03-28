"""
Convert Musical_Instruments_5.json to JSONL format.
Run this once after downloading the dataset.

Usage:
    python src/convert_to_jsonl.py
"""

import json
from pathlib import Path

INPUT  = "data/Musical_Instruments_5.json"
OUTPUT = "data/amazon_musical_instruments_reviews.jsonl"

count = 0
with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        review = data.get("reviewText", "")
        if review.strip():
            record = {
                "id":      idx,
                "asin":    data.get("asin", ""),
                "rating":  data.get("overall", None),
                "summary": data.get("summary", ""),
                "review":  review,
            }
            fout.write(json.dumps(record) + "\n")
            count += 1

print(f"Converted {count:,} reviews → {OUTPUT}")
