"""Small I/O helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from PIL import Image


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlAppender:
    """Append results one by one so that interrupted runs can resume."""

    def __init__(self, path: str | Path, key: str = "id"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.done = set()
        if self.path.exists():
            for r in iter_jsonl(self.path):
                self.done.add(str(r[key]))
        self._f = open(self.path, "a", encoding="utf-8")

    def __contains__(self, item) -> bool:
        return str(item) in self.done

    def write(self, row: Dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        self.done.add(str(row[self.key]))

    def close(self) -> None:
        self._f.close()


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_lines(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]
