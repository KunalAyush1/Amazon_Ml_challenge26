"""Build feature-based training pair datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from business_entity_resol.datasets.negative_sampling import (
    sample_negative_pairs,
)
from business_entity_resol.datasets.positive_pairs import (
    generate_positive_pairs,
)
from business_entity_resol.features.pair_features import pair_features


def build_pair_dataset(
    source1_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    ground_truth: Mapping[str, Iterable[str]],
    candidate_lookup: Mapping[str, Mapping[str, Any]],
    *,
    negative_ratio: int = 1,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build labeled candidate pairs with pairwise features.

    ``candidate_records`` describes the candidate pool and must contain:
        source1_entity_id
        candidate_entity_id

    ``candidate_lookup`` maps candidate entity IDs to their normalized
    S2/S3 records.
    """

    source1_by_id = {
        str(record["entity_id"]): record
        for record in source1_records
    }

    candidate_list = list(candidate_records)

    candidate_mapping: dict[str, set[str]] = {}
    candidate_metadata: dict[
        tuple[str, str],
        Mapping[str, Any],
    ] = {}

    for candidate in candidate_list:
        source1_id = str(candidate["source1_entity_id"])
        candidate_id = str(candidate["candidate_entity_id"])

        candidate_mapping.setdefault(
            source1_id,
            set(),
        ).add(candidate_id)

        candidate_metadata[
            (source1_id, candidate_id)
        ] = candidate

    positive_pairs = list(
        generate_positive_pairs_from_mapping(ground_truth)
    )

    negative_pairs = sample_negative_pairs(
        candidate_mapping,
        ground_truth,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
    )

    rows: list[dict[str, Any]] = []

    for pair in positive_pairs + negative_pairs:
        source1_id = str(pair["source1_entity_id"])
        candidate_id = str(pair["candidate_entity_id"])

        source1 = source1_by_id.get(source1_id)
        candidate = candidate_lookup.get(candidate_id)
        metadata = candidate_metadata.get(
            (source1_id, candidate_id)
        )

        # Only build training examples when the pair exists in the
        # candidate pool and both records are available.
        if source1 is None or candidate is None or metadata is None:
            continue

        features = pair_features(
            dict(source1),
            dict(candidate),
        )

        row: dict[str, Any] = {
            "source1_entity_id": source1_id,
            "candidate_entity_id": candidate_id,
            "candidate_source": metadata.get("source"),
            "label": int(pair["label"]),
        }

        row.update(features)

        rows.append(row)

    return rows


def generate_positive_pairs_from_mapping(
    ground_truth: Mapping[str, Iterable[str]],
) -> Iterable[dict[str, str | int]]:
    """Generate positive pair rows from an in-memory ground-truth mapping."""

    for source1_id, matched_ids in ground_truth.items():
        for candidate_id in matched_ids:
            if candidate_id:
                yield {
                    "source1_entity_id": str(source1_id),
                    "candidate_entity_id": str(candidate_id),
                    "label": 1,
                }