from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def stable_int(parts: tuple[str, ...]) -> int:
    joined = "||".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()
    return int(digest[:16], 16)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_path(config_path: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.5 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]
