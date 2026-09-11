"""Structured localization results; all geometry uses source-image pixels."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class STLResult:
    answer: str
    boxes: list[dict[str, float]]
    points: list[dict[str, float]]
    image_size: tuple[int, int]
    task: str
    query: str | None
    stats: dict[str, Any] | None = None
