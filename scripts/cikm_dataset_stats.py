#!/usr/bin/env python
"""Recompute Table 2 (dataset statistics) directly from the processed splits.

The statistics in the paper were originally recorded during preprocessing. This
script recovers them from the shipped parquet files so a reader can verify the
table without rerunning preprocessing, and so the table cannot silently drift
from the data actually released.

Definitions match the paper:
  users         distinct users in train + val + test
  |I|           distinct items appearing in the TRAINING split (the retrieval
                catalog); test-only items are invisible to retrieval
  interactions  rows in train + val + test
  density       (interactions / users) / |I|

Usage:  python scripts/cikm_dataset_stats.py
"""

import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(REPO, "data", "processed")

DATASETS = [
    ("Beauty", "amazon_beauty_sampled", "Product"),
    ("Movies", "amazon_movies_sampled", "Product"),
    ("Electronics", "amazon_electronics_sampled", "Product"),
    ("Sports", "amazon_sports_sampled", "Product"),
    ("Toys", "amazon_toys_sampled", "Product"),
    ("Office", "amazon_office_sampled", "Product"),
    ("MovieLens-25M", "movielens_25m", "Movie"),
    ("MIND News", "mind_news", "News"),
]

# What the camera-ready prints, for a regression check.
PAPER = {
    "Beauty":        (938, 1416, 8241, 0.62),
    "Movies":        (1998, 53709, 358587, 0.33),
    "Electronics":   (2000, 28981, 183913, 0.32),
    "Sports":        (1580, 5246, 37961, 0.46),
    "Toys":          (1594, 9545, 69574, 0.46),
    "Office":        (1704, 5037, 40625, 0.47),
    "MovieLens-25M": (10000, 19894, 3949023, 1.99),
    "MIND News":     (86746, 20213, 2278768, 0.13),
}


def load(split_dir, name):
    path = os.path.join(split_dir, f"{name}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def stats(dirname):
    d = os.path.join(PROC, dirname)
    parts = [load(d, s) for s in ("train", "val", "test")]
    train = parts[0]
    if train is None:
        return None
    allrows = pd.concat([p for p in parts if p is not None], ignore_index=True)
    users = allrows["user_id"].nunique()
    catalog = train["item_id"].nunique()
    inter = len(allrows)
    density = 100.0 * (inter / users) / catalog
    # Items that exist only outside the training split are unrecoverable by
    # retrieval; the paper calls this the open-catalog setting.
    unseen = len(set(allrows["item_id"]) - set(train["item_id"]))
    return users, catalog, inter, density, unseen


def main():
    rows, mismatches = [], []
    for label, dirname, domain in DATASETS:
        s = stats(dirname)
        if s is None:
            print(f"[skip] {label}: {dirname} not present")
            continue
        users, catalog, inter, density, unseen = s
        rows.append((label, users, catalog, inter, density, domain, unseen))
        exp = PAPER.get(label)
        if exp:
            got = (users, catalog, inter, round(density, 2))
            if got != (exp[0], exp[1], exp[2], exp[3]):
                mismatches.append((label, exp, got))

    hdr = f"{'Dataset':<15}{'Users':>8}{'|I|':>8}{'Interactions':>14}" \
          f"{'Density':>9}  {'Domain':<8}{'open-cat. items':>16}"
    print(hdr)
    print("-" * len(hdr))
    for label, users, catalog, inter, density, domain, unseen in rows:
        print(f"{label:<15}{users:>8,}{catalog:>8,}{inter:>14,}"
              f"{density:>8.2f}%  {domain:<8}{unseen:>16,}")

    print()
    if mismatches:
        print("MISMATCH vs the camera-ready table:")
        for label, exp, got in mismatches:
            print(f"  {label}: paper={exp} computed={got}")
    else:
        print("All present datasets match the camera-ready Table 2.")

    out = os.path.join(REPO, "experiments", "logs", "cikm_dataset_stats.json")
    with open(out, "w") as fh:
        json.dump({"datasets": [
            {"name": r[0], "users": r[1], "catalog_items": r[2],
             "interactions": r[3], "density_pct": round(r[4], 4),
             "domain": r[5], "open_catalog_items": r[6]} for r in rows]},
            fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
