"""Build a model-ready feature dataframe from candidate pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from business_entity_resol.features.pair_features import pair_features


ID_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
)


def build_candidate_feature_dataframe(
    source1_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    candidate_lookup: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build one feature row for every candidate pair.

    Parameters
    ----------
    source1_records:
        Normalized Source-1 records. Each record must contain
        ``entity_id``.

    candidate_records:
        CandidateUnion records. Each record should contain:
            source1_entity_id
            candidate_entity_id
            source

        It may also contain blocker provenance information:
            blocker_provenance
            ranks
            num_blockers
            best_rank

    candidate_lookup:
        Mapping from candidate entity ID to its normalized
        Source-2/Source-3 record.

    Returns
    -------
    pandas.DataFrame
        One row per candidate pair containing:
            - pair identifiers
            - source indicator features
            - all pairwise model features
    """
    source1_by_id = {
        str(record["entity_id"]): dict(record)
        for record in source1_records
    }

    rows: list[dict[str, Any]] = []

    for metadata in candidate_records:
        source1_id = str(
            metadata["source1_entity_id"]
        )
        candidate_id = str(
            metadata["candidate_entity_id"]
        )

        source1 = source1_by_id.get(source1_id)
        candidate = candidate_lookup.get(candidate_id)

        if source1 is None:
            raise KeyError(
                f"Source-1 entity not found: {source1_id}"
            )

        if candidate is None:
            raise KeyError(
                f"Candidate entity not found: {candidate_id}"
            )

        # Merge blocker metadata into the candidate record so that
        # blocker_features() can use it.
        candidate_for_features = {
            **dict(candidate),
            **dict(metadata),
        }

        features = pair_features(
            source1,
            candidate_for_features,
        )

        candidate_source = str(
            metadata.get("source", "")
        )

        row: dict[str, Any] = {
            "source1_entity_id": source1_id,
            "candidate_entity_id": candidate_id,
            "candidate_source": candidate_source,
            "candidate_is_s2": int(
                candidate_source == "S2"
            ),
            "candidate_is_s3": int(
                candidate_source == "S3"
            ),
        }

        row.update(features)

        rows.append(row)

    return pd.DataFrame(rows)


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return columns suitable for model training/inference.

    Identifier and metadata columns are excluded. The returned
    columns should contain only numeric model features.
    """
    excluded = {
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "label",
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]