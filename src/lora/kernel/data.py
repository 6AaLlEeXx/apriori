from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    prompt: str
    completion: str


def load_pair_split(path: str | Path, split: str) -> list[PairRecord]:
    records: list[PairRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                PairRecord(
                    pair_id=f"{split}-{index:06d}",
                    split=split,
                    prompt=str(payload["prompt"]),
                    completion=str(payload["completion"]),
                )
            )
    return records


def maybe_subset_pairs(
    records: list[PairRecord],
    limit: int,
    seed: int,
) -> list[PairRecord]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=limit, replace=False))
    return [records[int(index)] for index in indices]
