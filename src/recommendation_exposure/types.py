from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ItemRecord:
    item_id: str
    title: str
    category: str
    description: str
    tags: str
    text: str


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    group: str
    secondary_group: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Candidate:
    item_id: str
    label: int


@dataclass(frozen=True, slots=True)
class UserEvalCase:
    split: str
    user_id: str
    group: str
    history_item_ids: tuple[str, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    item_id: str
    label: int
    score: float
    item_group: str | None = None


@dataclass(frozen=True, slots=True)
class ScoredUserEvalCase:
    split: str
    user_id: str
    group: str
    history_item_ids: tuple[str, ...]
    candidates: tuple[ScoredCandidate, ...]


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    name: str
    users: dict[str, UserRecord]
    items: dict[str, ItemRecord]
    train_histories: dict[str, tuple[str, ...]]
    calibration_cases: tuple[UserEvalCase, ...]
    test_cases: tuple[UserEvalCase, ...]
    train_popularity: dict[str, int]
    config_path: Path
