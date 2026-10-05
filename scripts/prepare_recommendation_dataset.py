#!/usr/bin/env python3
"""Normalize raw recommendation datasets into the experiment CSV contract."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure.utils import load_json


MANIFEST_PATH = ESER_ROOT / "experiments" / "recommendation_eser" / "assets_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["movielens1m", "amazon_books", "lastfm1k", "bookcrossing"])
    parser.add_argument("--max-users", type=int, default=None, help="Optional cap after sorting by interaction count.")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument(
        "--prepared-suffix",
        default="",
        help="Optional suffix appended to the prepared dataset directory name, for example 'sample200'.",
    )
    return parser.parse_args()


def _manifest() -> dict:
    return load_json(MANIFEST_PATH)


def _ensure_dirs(dataset_id: str, prepared_suffix: str = "") -> tuple[Path, Path]:
    manifest = _manifest()["datasets"][dataset_id]
    raw_dir = (ESER_ROOT.parent / manifest["raw_dir"]).resolve()
    prepared_dir = (ESER_ROOT.parent / manifest["prepared_dir"]).resolve()
    if prepared_suffix:
        prepared_dir = prepared_dir.with_name(f"{prepared_dir.name}_{prepared_suffix}")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, prepared_dir


def _activity_group(count: int) -> str:
    if count <= 10:
        return "low"
    if count <= 30:
        return "medium"
    return "high"


def _region_group(country: str) -> str:
    lower = country.strip().lower()
    mapping = {
        "united states": "north_america",
        "canada": "north_america",
        "mexico": "north_america",
        "united kingdom": "europe",
        "germany": "europe",
        "france": "europe",
        "spain": "europe",
        "italy": "europe",
        "poland": "europe",
        "sweden": "europe",
        "norway": "europe",
        "finland": "europe",
        "netherlands": "europe",
        "belgium": "europe",
        "austria": "europe",
        "switzerland": "europe",
        "russia": "europe",
        "brazil": "south_america",
        "argentina": "south_america",
        "chile": "south_america",
        "australia": "oceania",
        "new zealand": "oceania",
        "japan": "asia",
        "china": "asia",
        "india": "asia",
        "south korea": "asia",
        "taiwan": "asia",
        "hong kong": "asia",
        "singapore": "asia",
    }
    return mapping.get(lower, "other")


def _cold_start_group(count: int) -> str:
    return "cold_start" if count <= 8 else "warm"


def _age_group(age_value: str) -> str:
    try:
        age = int(float(age_value))
    except Exception:
        return "unknown"
    if age < 25:
        return "under_25"
    if age < 45:
        return "25_44"
    return "45_plus"


def _lastfm_region_binary_group(region_group: str) -> str:
    return "europe" if region_group == "europe" else "non_europe"


def _bookcrossing_age_binary_group(age_group: str) -> str:
    return "under_45" if age_group in {"under_25", "25_44"} else "45_plus_or_unknown"


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_movielens1m(args: argparse.Namespace) -> dict:
    raw_dir, prepared_dir = _ensure_dirs("movielens1m", args.prepared_suffix)
    extract_dir = raw_dir / "ml-1m"
    users_path = extract_dir / "users.dat"
    movies_path = extract_dir / "movies.dat"
    ratings_path = extract_dir / "ratings.dat"

    movies = {}
    with movies_path.open(encoding="latin-1") as handle:
        for line in handle:
            movie_id, title, genres = line.rstrip("\n").split("::")
            movies[movie_id] = {
                "item_id": movie_id,
                "title": title,
                "category": genres.replace("|", ", "),
                "description": "",
                "tags": genres,
            }

    user_counts: Counter[str] = Counter()
    ratings_rows = []
    with ratings_path.open(encoding="utf-8") as handle:
        for line in handle:
            user_id, item_id, rating, timestamp = line.rstrip("\n").split("::")
            ratings_rows.append((user_id, item_id, timestamp))
            user_counts[user_id] += 1

    allowed_users = set(sorted(user_counts, key=lambda user_id: (-user_counts[user_id], user_id)))
    if args.max_users is not None:
        allowed_users = set(list(allowed_users)[: args.max_users])

    users = []
    with users_path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, gender, age, occupation, zip_code = line.rstrip("\n").split("::")
            if user_id not in allowed_users:
                continue
            users.append(
                {
                    "user_id": user_id,
                    "gender_group": gender,
                    "activity_group": _activity_group(user_counts[user_id]),
                    "age_bucket": age,
                    "occupation": occupation,
                    "zip_code": zip_code,
                }
            )

    interactions = [
        {"user_id": user_id, "item_id": item_id, "timestamp": timestamp}
        for user_id, item_id, timestamp in ratings_rows
        if user_id in allowed_users
    ]

    _write_csv(prepared_dir / "users.csv", ["user_id", "gender_group", "activity_group", "age_bucket", "occupation", "zip_code"], users)
    _write_csv(prepared_dir / "items.csv", ["item_id", "title", "category", "description", "tags"], movies.values())
    _write_csv(prepared_dir / "interactions.csv", ["user_id", "item_id", "timestamp"], interactions)
    return {"prepared_dir": str(prepared_dir), "num_users": len(users), "num_items": len(movies), "num_interactions": len(interactions)}


def _iter_json_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def prepare_amazon_books(args: argparse.Namespace) -> dict:
    raw_dir, prepared_dir = _ensure_dirs("amazon_books", args.prepared_suffix)
    reviews_path = raw_dir / "Books_5.json.gz"
    meta_path = raw_dir / "meta_Books.json.gz"

    interactions = []
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for payload in _iter_json_gz(reviews_path):
        user_id = payload["reviewerID"]
        item_id = payload["asin"]
        timestamp = str(int(payload.get("unixReviewTime", 0)))
        interactions.append((user_id, item_id, timestamp))
        user_counts[user_id] += 1
        item_counts[item_id] += 1

    filtered = [
        row for row in interactions
        if user_counts[row[0]] >= args.min_user_interactions and item_counts[row[1]] >= args.min_item_interactions
    ]
    filtered_user_counts: Counter[str] = Counter(row[0] for row in filtered)
    allowed_users = [user_id for user_id, _ in sorted(filtered_user_counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    if args.max_users is not None:
        allowed_users = allowed_users[: args.max_users]
    allowed_user_set = set(allowed_users)
    filtered = [row for row in filtered if row[0] in allowed_user_set]

    kept_items = {item_id for _, item_id, _ in filtered}
    items = {}
    for payload in _iter_json_gz(meta_path):
        item_id = payload.get("asin")
        if item_id not in kept_items:
            continue
        categories = payload.get("category") or payload.get("categories") or []
        if isinstance(categories, list):
            flat_categories = []
            for entry in categories:
                if isinstance(entry, list):
                    flat_categories.extend(str(part) for part in entry)
                else:
                    flat_categories.append(str(entry))
            category_text = " | ".join(flat_categories[:5])
        else:
            category_text = str(categories)
        description = payload.get("description")
        if isinstance(description, list):
            description = " ".join(str(part) for part in description[:3])
        feature = payload.get("feature")
        if isinstance(feature, list):
            feature = " ".join(str(part) for part in feature[:3])
        items[item_id] = {
            "item_id": item_id,
            "title": str(payload.get("title", item_id)),
            "category": category_text,
            "description": " ".join(part for part in [str(description or ""), str(feature or "")] if part).strip(),
            "tags": category_text,
        }

    users = []
    for user_id in allowed_users:
        count = filtered_user_counts[user_id]
        users.append(
            {
                "user_id": user_id,
                "cold_start_group": _cold_start_group(count),
                "activity_group": _activity_group(count),
                "interaction_count": count,
            }
        )

    interaction_rows = [
        {"user_id": user_id, "item_id": item_id, "timestamp": timestamp}
        for user_id, item_id, timestamp in filtered
        if item_id in items
    ]

    _write_csv(prepared_dir / "users.csv", ["user_id", "cold_start_group", "activity_group", "interaction_count"], users)
    _write_csv(prepared_dir / "items.csv", ["item_id", "title", "category", "description", "tags"], items.values())
    _write_csv(prepared_dir / "interactions.csv", ["user_id", "item_id", "timestamp"], interaction_rows)
    return {"prepared_dir": str(prepared_dir), "num_users": len(users), "num_items": len(items), "num_interactions": len(interaction_rows)}


def prepare_lastfm1k(args: argparse.Namespace) -> dict:
    raw_dir, prepared_dir = _ensure_dirs("lastfm1k", args.prepared_suffix)
    extract_dir = raw_dir / "lastfm-dataset-1K"
    interactions_path = extract_dir / "userid-timestamp-artid-artname-traid-traname.tsv"
    profile_path = extract_dir / "userid-profile.tsv"

    profiles = {}
    with profile_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            user_id, gender, age, country, signup = parts[:5]
            profiles[user_id] = {"gender": gender, "age": age, "country": country, "signup": signup}

    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    interactions = []
    item_meta = {}
    with interactions_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            user_id, timestamp, artist_id, artist_name, track_id, track_name = parts[:6]
            item_id = track_id or f"track::{artist_name}::{track_name}"
            interactions.append((user_id, item_id, artist_name, track_name, timestamp))
            user_counts[user_id] += 1
            item_counts[item_id] += 1
            item_meta[item_id] = {
                "item_id": item_id,
                "title": track_name,
                "category": "music",
                "description": artist_name,
                "tags": artist_name,
            }

    filtered = [
        row for row in interactions
        if user_counts[row[0]] >= args.min_user_interactions and item_counts[row[1]] >= args.min_item_interactions and row[0] in profiles
    ]
    filtered_user_counts: Counter[str] = Counter(row[0] for row in filtered)
    allowed_users = [user_id for user_id, _ in sorted(filtered_user_counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    if args.max_users is not None:
        allowed_users = allowed_users[: args.max_users]
    allowed_user_set = set(allowed_users)
    filtered = [row for row in filtered if row[0] in allowed_user_set]
    kept_items = {row[1] for row in filtered}

    users = []
    for user_id in allowed_users:
        profile = profiles[user_id]
        count = filtered_user_counts[user_id]
        region_group = _region_group(profile["country"])
        users.append(
            {
                "user_id": user_id,
                "region_group": region_group,
                "region_binary_group": _lastfm_region_binary_group(region_group),
                "activity_group": _activity_group(count),
                "country": profile["country"],
                "gender": profile["gender"],
                "age": profile["age"],
            }
        )

    items = [item_meta[item_id] for item_id in sorted(kept_items)]
    interaction_rows = [{"user_id": row[0], "item_id": row[1], "timestamp": row[4]} for row in filtered]

    _write_csv(
        prepared_dir / "users.csv",
        ["user_id", "region_group", "region_binary_group", "activity_group", "country", "gender", "age"],
        users,
    )
    _write_csv(prepared_dir / "items.csv", ["item_id", "title", "category", "description", "tags"], items)
    _write_csv(prepared_dir / "interactions.csv", ["user_id", "item_id", "timestamp"], interaction_rows)
    return {"prepared_dir": str(prepared_dir), "num_users": len(users), "num_items": len(items), "num_interactions": len(interaction_rows)}


def prepare_bookcrossing(args: argparse.Namespace) -> dict:
    raw_dir, prepared_dir = _ensure_dirs("bookcrossing", args.prepared_suffix)
    extract_dir = raw_dir / "BX-CSV-Dump"
    if not extract_dir.exists():
        extract_dir = raw_dir
    users_path = extract_dir / "BX-Users.csv"
    books_path = extract_dir / "BX-Books.csv"
    ratings_path = extract_dir / "BX-Book-Ratings.csv"

    users_meta = {}
    with users_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            user_id = row["User-ID"].strip('"')
            users_meta[user_id] = {
                "location": row["Location"].strip('"'),
                "age": row["Age"].strip('"'),
            }

    books = {}
    with books_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            item_id = row["ISBN"].strip('"')
            books[item_id] = {
                "item_id": item_id,
                "title": row["Book-Title"].strip('"'),
                "category": row["Publisher"].strip('"'),
                "description": " ".join(
                    part for part in [
                        row["Book-Author"].strip('"'),
                        row["Year-Of-Publication"].strip('"'),
                        row["Publisher"].strip('"'),
                    ] if part
                ),
                "tags": row["Book-Author"].strip('"'),
            }

    interactions = []
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    with ratings_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            user_id = row["User-ID"].strip('"')
            item_id = row["ISBN"].strip('"')
            rating = row["Book-Rating"].strip('"')
            interactions.append((user_id, item_id, rating))
            user_counts[user_id] += 1
            item_counts[item_id] += 1

    filtered = [
        row for row in interactions
        if row[0] in users_meta and row[1] in books and user_counts[row[0]] >= args.min_user_interactions and item_counts[row[1]] >= args.min_item_interactions
    ]
    filtered_user_counts: Counter[str] = Counter(row[0] for row in filtered)
    allowed_users = [user_id for user_id, _ in sorted(filtered_user_counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    if args.max_users is not None:
        allowed_users = allowed_users[: args.max_users]
    allowed_user_set = set(allowed_users)
    filtered = [row for row in filtered if row[0] in allowed_user_set]
    kept_items = {row[1] for row in filtered}

    users = []
    for user_id in allowed_users:
        meta = users_meta[user_id]
        count = filtered_user_counts[user_id]
        age_group = _age_group(meta["age"])
        users.append(
            {
                "user_id": user_id,
                "age_group": age_group,
                "age_binary_group": _bookcrossing_age_binary_group(age_group),
                "activity_group": _activity_group(count),
                "location": meta["location"],
                "age": meta["age"],
            }
        )

    item_rows = [books[item_id] for item_id in sorted(kept_items)]
    interaction_rows = [
        {"user_id": user_id, "item_id": item_id, "timestamp": index + 1}
        for index, (user_id, item_id, _rating) in enumerate(filtered)
    ]

    _write_csv(
        prepared_dir / "users.csv",
        ["user_id", "age_group", "age_binary_group", "activity_group", "location", "age"],
        users,
    )
    _write_csv(prepared_dir / "items.csv", ["item_id", "title", "category", "description", "tags"], item_rows)
    _write_csv(prepared_dir / "interactions.csv", ["user_id", "item_id", "timestamp"], interaction_rows)
    return {"prepared_dir": str(prepared_dir), "num_users": len(users), "num_items": len(item_rows), "num_interactions": len(interaction_rows)}


def main() -> None:
    args = parse_args()
    if args.dataset == "movielens1m":
        payload = prepare_movielens1m(args)
    elif args.dataset == "amazon_books":
        payload = prepare_amazon_books(args)
    elif args.dataset == "lastfm1k":
        payload = prepare_lastfm1k(args)
    else:
        payload = prepare_bookcrossing(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
