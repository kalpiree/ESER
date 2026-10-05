from __future__ import annotations

import csv
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .types import Candidate, ItemRecord, PreparedDataset, UserEvalCase, UserRecord
from .utils import resolve_path, stable_int


# Long item descriptions can exceed the default CSV parser field limit.
csv.field_size_limit(sys.maxsize)


def _load_items(path: Path) -> dict[str, ItemRecord]:
    items: dict[str, ItemRecord] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = row["item_id"]
            title = row.get("title", "").strip()
            category = row.get("category", "").strip()
            description = row.get("description", "").strip()
            tags = row.get("tags", "").strip()
            text = " ".join(part for part in [title, category, description, tags] if part).strip()
            items[item_id] = ItemRecord(
                item_id=item_id,
                title=title,
                category=category,
                description=description,
                tags=tags,
                text=text,
            )
            row_count += 1
            if row_count % 100_000 == 0:
                print(f"[rec:data] loaded_items={row_count}", flush=True)
    return items


def _load_users(
    path: Path,
    group_column: str,
    secondary_group_column: str | None,
    group_value_map: dict[str, str] | None = None,
    numeric_group_cfg: dict[str, object] | None = None,
) -> dict[str, UserRecord]:
    users: dict[str, UserRecord] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = row["user_id"]
            attributes = {key: value for key, value in row.items() if key != "user_id"}
            group = row[group_column]
            if numeric_group_cfg:
                numeric_column = str(numeric_group_cfg["column"])
                threshold = float(numeric_group_cfg["threshold"])
                lower_label = str(numeric_group_cfg.get("lower_label", "lower"))
                upper_label = str(numeric_group_cfg.get("upper_label", "upper"))
                try:
                    numeric_value = float(row.get(numeric_column, "0") or 0)
                except ValueError:
                    numeric_value = 0.0
                group = upper_label if numeric_value >= threshold else lower_label
            if group_value_map:
                group = group_value_map.get(group, group)
            users[user_id] = UserRecord(
                user_id=user_id,
                group=group,
                secondary_group=row.get(secondary_group_column) if secondary_group_column else None,
                attributes=attributes,
            )
            row_count += 1
            if row_count % 100_000 == 0:
                print(f"[rec:data] loaded_users={row_count}", flush=True)
    return users


def _load_interactions(path: Path, timestamp_column: str) -> dict[str, list[tuple[float, str]]]:
    interactions: dict[str, list[tuple[float, str]]] = defaultdict(list)
    row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            interactions[row["user_id"]].append((_parse_timestamp(row[timestamp_column]), row["item_id"]))
            row_count += 1
            if row_count % 500_000 == 0:
                print(f"[rec:data] loaded_interactions={row_count}", flush=True)
    for user_id, values in interactions.items():
        values.sort(key=lambda pair: (pair[0], pair[1]))
    return interactions


def _parse_timestamp(raw_value: str) -> float:
    value = raw_value.strip()
    try:
        return float(value)
    except ValueError:
        pass

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {raw_value!r}") from exc


def _sample_negatives(
    all_item_ids: list[str],
    seen_items: set[str],
    positive_item_id: str,
    sample_size: int,
    seed_parts: tuple[str, ...],
) -> list[str]:
    rng = random.Random(stable_int(seed_parts))
    unavailable = set(seen_items)
    unavailable.add(positive_item_id)
    available_count = max(0, len(all_item_ids) - len(unavailable))
    if sample_size >= available_count:
        negatives = [item_id for item_id in all_item_ids if item_id not in unavailable]
        rng.shuffle(negatives)
        return negatives

    # Avoid rebuilding the full unseen-item pool for every user on large datasets.
    negatives: list[str] = []
    picked: set[str] = set()
    if available_count >= sample_size * 4:
        while len(negatives) < sample_size:
            candidate = all_item_ids[rng.randrange(len(all_item_ids))]
            if candidate in unavailable or candidate in picked:
                continue
            picked.add(candidate)
            negatives.append(candidate)
        return negatives

    pool = [item_id for item_id in all_item_ids if item_id not in unavailable]
    if sample_size >= len(pool):
        negatives = list(pool)
        rng.shuffle(negatives)
        return negatives
    return rng.sample(pool, sample_size)


def _build_case(
    *,
    split: str,
    user_id: str,
    group: str,
    history_item_ids: tuple[str, ...],
    positive_item_id: str,
    negatives: list[str],
) -> UserEvalCase:
    candidates = [Candidate(item_id=positive_item_id, label=1)]
    candidates.extend(Candidate(item_id=item_id, label=0) for item_id in negatives)
    rng = random.Random(stable_int((split, user_id, positive_item_id, "shuffle")))
    rng.shuffle(candidates)
    return UserEvalCase(
        split=split,
        user_id=user_id,
        group=group,
        history_item_ids=history_item_ids,
        candidates=tuple(candidates),
    )


def prepare_dataset(config: dict) -> PreparedDataset:
    config_path = Path(config["_config_path"])
    data_cfg = config["data"]
    split_cfg = config["split"]
    case_seed_name = str(data_cfg.get("case_seed_name", config["name"]))

    print(f"[rec:data] items_path={resolve_path(config_path, data_cfg['items_path'])}", flush=True)
    items = _load_items(resolve_path(config_path, data_cfg["items_path"]))
    print(f"[rec:data] users_path={resolve_path(config_path, data_cfg['users_path'])}", flush=True)
    users = _load_users(
        resolve_path(config_path, data_cfg["users_path"]),
        data_cfg["group_column"],
        data_cfg.get("secondary_group_column"),
        data_cfg.get("group_value_map"),
        data_cfg.get("numeric_group"),
    )
    print(f"[rec:data] interactions_path={resolve_path(config_path, data_cfg['interactions_path'])}", flush=True)
    interactions = _load_interactions(
        resolve_path(config_path, data_cfg["interactions_path"]),
        data_cfg.get("timestamp_column", "timestamp"),
    )

    min_history = int(data_cfg.get("min_history", 3))
    sample_fraction = float(data_cfg.get("user_sample_fraction", 1.0))
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError("data.user_sample_fraction must be in the interval (0, 1].")
    max_eval_users_cfg = data_cfg.get("max_eval_users", os.environ.get("REC_MAX_EVAL_USERS"))
    max_eval_users = int(max_eval_users_cfg) if max_eval_users_cfg not in (None, "") else None
    calibration_positive_count = int(split_cfg.get("calibration_positive_count", 1))
    test_positive_count = int(split_cfg.get("test_positive_count", 1))
    if calibration_positive_count != 1 or test_positive_count != 1:
        raise ValueError("The current pipeline supports exactly one calibration and one test positive per user.")

    negative_sample_size = int(split_cfg.get("negative_sample_size", 50))
    negative_override = os.environ.get("REC_NEGATIVE_SAMPLE_SIZE_OVERRIDE")
    if negative_override is not None:
        negative_sample_size = int(negative_override)
        print(f"[rec:data] override_negative_sample_size={negative_sample_size}", flush=True)
    all_item_ids = sorted(items)

    calibration_cases: list[UserEvalCase] = []
    test_cases: list[UserEvalCase] = []
    train_histories: dict[str, tuple[str, ...]] = {}
    train_popularity: Counter[str] = Counter()
    minimum_interactions = min_history + calibration_positive_count + test_positive_count

    kept_users = 0
    scanned_eligible_users = 0
    for user_id, user_record in sorted(users.items()):
        user_interactions = interactions.get(user_id, [])
        if len(user_interactions) < minimum_interactions:
            continue
        scanned_eligible_users += 1
        if sample_fraction < 1.0:
            bucket = stable_int((case_seed_name, user_id, "user_sample")) / float(2**64 - 1)
            if bucket >= sample_fraction:
                continue
        if max_eval_users is not None and kept_users >= max_eval_users:
            continue

        ordered_item_ids = [item_id for _, item_id in user_interactions]
        train_cutoff = len(ordered_item_ids) - calibration_positive_count - test_positive_count
        train_history = tuple(ordered_item_ids[:train_cutoff])
        calibration_positive = ordered_item_ids[train_cutoff]
        test_positive = ordered_item_ids[-1]
        full_seen = set(ordered_item_ids)

        train_histories[user_id] = train_history
        train_popularity.update(train_history)

        calibration_negatives = _sample_negatives(
            all_item_ids=all_item_ids,
            seen_items=full_seen,
            positive_item_id=calibration_positive,
            sample_size=negative_sample_size,
            seed_parts=(case_seed_name, user_id, "calibration"),
        )
        calibration_cases.append(
            _build_case(
                split="calibration",
                user_id=user_id,
                group=user_record.group,
                history_item_ids=train_history,
                positive_item_id=calibration_positive,
                negatives=calibration_negatives,
            )
        )

        test_history = tuple(ordered_item_ids[:-1])
        test_negatives = _sample_negatives(
            all_item_ids=all_item_ids,
            seen_items=full_seen,
            positive_item_id=test_positive,
            sample_size=negative_sample_size,
            seed_parts=(case_seed_name, user_id, "test"),
        )
        test_cases.append(
            _build_case(
                split="test",
                user_id=user_id,
                group=user_record.group,
                history_item_ids=test_history,
                positive_item_id=test_positive,
                negatives=test_negatives,
            )
        )
        kept_users += 1
        if kept_users % 50_000 == 0:
            print(
                "[rec:data] built_cases "
                f"users={kept_users} calibration_cases={len(calibration_cases)} test_cases={len(test_cases)}",
                flush=True,
            )

    if sample_fraction < 1.0 or max_eval_users is not None:
        print(
            "[rec:data] user_subset "
            f"eligible={scanned_eligible_users} kept={kept_users} "
            f"fraction={sample_fraction} max_eval_users={max_eval_users}",
            flush=True,
        )

    return PreparedDataset(
        name=config["name"],
        users=users,
        items=items,
        train_histories=train_histories,
        calibration_cases=tuple(calibration_cases),
        test_cases=tuple(test_cases),
        train_popularity=dict(train_popularity),
        config_path=config_path,
    )
