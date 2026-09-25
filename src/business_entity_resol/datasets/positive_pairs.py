"""Positive pair generation from ground-truth matches."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path


def generate_positive_pairs(
    ground_truth_path: str | Path,
) -> Iterator[dict[str, str | int]]:
    """Generate one positive pair for every ground-truth match.

    Parameters
    ----------
    ground_truth_path:
        Path to the training ground-truth TSV.

    Yields
    ------
    dict
        A dictionary containing:
        - source1_entity_id
        - candidate_entity_id
        - label

    Notes
    -----
    Empty ground-truth rows produce no pairs.
    Multiple matched entities for one Source-1 entity produce
    multiple positive pairs.
    """
    ground_truth_path = Path(ground_truth_path)

    with ground_truth_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        required_columns = {
            "source1_entity_id",
            "matched_entity_ids",
        }

        if not required_columns.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Ground-truth file must contain columns: "
                "source1_entity_id and matched_entity_ids"
            )

        for row in reader:
            source1_entity_id = (row["source1_entity_id"] or "").strip()
            matched_entity_ids = (row["matched_entity_ids"] or "").strip()

            if not source1_entity_id:
                continue

            if not matched_entity_ids or matched_entity_ids == "-":
                continue

            for candidate_entity_id in matched_entity_ids.split(","):
                candidate_entity_id = candidate_entity_id.strip()

                if not candidate_entity_id:
                    continue

                yield {
                    "source1_entity_id": source1_entity_id,
                    "candidate_entity_id": candidate_entity_id,
                    "label": 1,
                }