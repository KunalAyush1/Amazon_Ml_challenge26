from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.io.ground_truth import (
    read_ground_truth,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_DIR = PROJECT_ROOT / "data" / "train"

WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "submission2_work"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)

EXISTING_CANDIDATES = (
    ARTIFACT_DIR
    / "submission2_candidates_2000.parquet"
)

PREFIX_CANDIDATES = (
    ARTIFACT_DIR
    / "submission2_prefix_candidates_2000.parquet"
)

COMBINED_CANDIDATES = (
    ARTIFACT_DIR
    / "submission2_combined_candidates_2000.parquet"
)

S1_SAMPLE = 2_000

# Prefix lengths to test.
NAME_PREFIX_LENGTH = 5
ADDRESS_PREFIX_LENGTH = 8


# ---------------------------------------------------------------------
# Deterministic S1 sample
# ---------------------------------------------------------------------


def get_sample_s1_ids(
    con: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Recover the exact 2,000-S1 sample used previously."""

    s1_glob = str(
        WORK_DIR
        / "train_source1"
        / "part_*.parquet"
    )

    rows = con.execute(
        f"""
        SELECT entity_id
        FROM read_parquet('{s1_glob}')
        ORDER BY hash(entity_id)
        LIMIT {S1_SAMPLE};
        """
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


# ---------------------------------------------------------------------
# Candidate recall
# ---------------------------------------------------------------------


def evaluate_recall(
    candidate_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    sample_s1_ids: list[str],
) -> tuple[float, float]:
    """Calculate link recall and complete-entity recall."""

    candidates_by_s1: dict[
        str,
        set[str],
    ] = {}

    for row in candidate_df.itertuples(
        index=False
    ):
        candidates_by_s1.setdefault(
            str(row.source1_entity_id),
            set(),
        ).add(
            str(row.candidate_entity_id)
        )

    total_true_links = 0
    retrieved_true_links = 0

    eligible_entities = 0
    complete_entities = 0

    for source1_id in sample_s1_ids:

        true_set = ground_truth.get(
            source1_id,
            set(),
        )

        if not true_set:
            continue

        eligible_entities += 1

        candidate_set = candidates_by_s1.get(
            source1_id,
            set(),
        )

        total_true_links += len(
            true_set
        )

        retrieved_true_links += len(
            true_set & candidate_set
        )

        if true_set.issubset(
            candidate_set
        ):
            complete_entities += 1

    link_recall = (
        retrieved_true_links
        / total_true_links
        if total_true_links
        else 0.0
    )

    complete_recall = (
        complete_entities
        / eligible_entities
        if eligible_entities
        else 0.0
    )

    return (
        link_recall,
        complete_recall,
    )


# ---------------------------------------------------------------------
# Candidate statistics
# ---------------------------------------------------------------------


def print_statistics(
    candidate_df: pd.DataFrame,
    sample_s1_ids: list[str],
) -> None:
    """Print candidate-count statistics."""

    counts = (
        candidate_df
        .groupby(
            "source1_entity_id"
        )
        .size()
        .reindex(
            sample_s1_ids,
            fill_value=0,
        )
    )

    print(
        f"  Mean candidates/S1: "
        f"{counts.mean():.3f}"
    )

    print(
        f"  Median candidates/S1: "
        f"{counts.median():.3f}"
    )

    print(
        f"  P95 candidates/S1: "
        f"{counts.quantile(0.95):.3f}"
    )

    print(
        f"  Max candidates/S1: "
        f"{counts.max():,}"
    )

    print(
        f"  Zero-candidate S1: "
        f"{(counts == 0).sum():,}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print("PREFIX BLOCKING EXPERIMENT — 2,000 S1")
    print("=" * 72)

    if not EXISTING_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Missing existing candidate artifact:\n"
            f"{EXISTING_CANDIDATES}"
        )

    print(
        "\nLoading ground truth..."
    )

    raw_ground_truth = read_ground_truth(
        TRAIN_DIR
        / "train_ground_truth.tsv"
    )

    ground_truth = {
        str(entity_id):
            set(matches or [])
        for (
            entity_id,
            matches,
        ) in raw_ground_truth.items()
    }

    con = duckdb.connect()

    con.execute(
        "PRAGMA threads=4"
    )

    # ---------------------------------------------------------------
    # Load existing 2,000-S1 candidates.
    # ---------------------------------------------------------------

    print(
        "\nLoading existing candidates..."
    )

    existing = pd.read_parquet(
        EXISTING_CANDIDATES
    )

    print(
        f"Existing candidates: "
        f"{len(existing):,}"
    )

    sample_s1_ids = get_sample_s1_ids(
        con
    )

    print(
        f"Sample S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    # ---------------------------------------------------------------
    # Create views over the already-normalized training data.
    # ---------------------------------------------------------------

    s1_glob = str(
        WORK_DIR
        / "train_source1"
        / "part_*.parquet"
    )

    s2_glob = str(
        WORK_DIR
        / "train_source2"
        / "part_*.parquet"
    )

    s3_glob = str(
        WORK_DIR
        / "train_source3"
        / "part_*.parquet"
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s1 AS

        SELECT
            entity_id,
            country,
            name_compact,
            address_compact

        FROM read_parquet(
            '{s1_glob}'
        )

        ORDER BY hash(entity_id)

        LIMIT {S1_SAMPLE};
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW targets AS

        SELECT
            entity_id AS candidate_entity_id,
            country,
            name_compact,
            address_compact

        FROM read_parquet(
            '{s2_glob}'
        )

        UNION ALL

        SELECT
            entity_id AS candidate_entity_id,
            country,
            name_compact,
            address_compact

        FROM read_parquet(
            '{s3_glob}'
        );
        """
    )

    # ---------------------------------------------------------------
    # Prefix candidates
    # ---------------------------------------------------------------

    print(
        "\nGenerating prefix candidates..."
    )

    prefix_query = f"""
        -- ---------------------------------------------------------
        -- Name prefix:
        -- same country + first 5 normalized compact name characters
        -- ---------------------------------------------------------

        SELECT
            s1.entity_id AS source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t

            ON s1.country <> ''
            AND s1.country = t.country

            AND s1.name_compact <> ''
            AND t.name_compact <> ''

            AND LEFT(
                s1.name_compact,
                {NAME_PREFIX_LENGTH}
            )
            =
            LEFT(
                t.name_compact,
                {NAME_PREFIX_LENGTH}
            )


        UNION


        -- ---------------------------------------------------------
        -- Address prefix:
        -- same country + first 8 normalized compact address chars
        -- ---------------------------------------------------------

        SELECT
            s1.entity_id AS source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t

            ON s1.country <> ''
            AND s1.country = t.country

            AND s1.address_compact <> ''
            AND t.address_compact <> ''

            AND LEFT(
                s1.address_compact,
                {ADDRESS_PREFIX_LENGTH}
            )
            =
            LEFT(
                t.address_compact,
                {ADDRESS_PREFIX_LENGTH}
            )
    """

    con.execute(
        f"""
        CREATE OR REPLACE TABLE prefix_candidates AS

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM (
            {prefix_query}
        );
        """
    )

    prefix_count = con.execute(
        """
        SELECT COUNT(*)
        FROM prefix_candidates;
        """
    ).fetchone()[0]

    print(
        f"Prefix-only candidates: "
        f"{prefix_count:,}"
    )

    prefix_df = con.execute(
        """
        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM prefix_candidates;
        """
    ).fetch_df()

    prefix_df.to_parquet(
        PREFIX_CANDIDATES,
        index=False,
    )

    # ---------------------------------------------------------------
    # Combined candidates
    # ---------------------------------------------------------------

    print(
        "\nCombining old + prefix candidates..."
    )

    combined = pd.concat(
        [
            existing[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ],
            prefix_df[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    combined.to_parquet(
        COMBINED_CANDIDATES,
        index=False,
    )

    print(
        f"Existing candidates: "
        f"{len(existing):,}"
    )

    print(
        f"Prefix candidates: "
        f"{len(prefix_df):,}"
    )

    print(
        f"Combined candidates: "
        f"{len(combined):,}"
    )

    # ---------------------------------------------------------------
    # Recall: existing
    # ---------------------------------------------------------------

    print(
        "\nExisting blocker recall:"
    )

    existing_link, existing_complete = (
        evaluate_recall(
            existing[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ],
            ground_truth,
            sample_s1_ids,
        )
    )

    print(
        f"  Link recall: "
        f"{existing_link:.6f}"
    )

    print(
        f"  Complete entity recall: "
        f"{existing_complete:.6f}"
    )

    print_statistics(
        existing[
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        ],
        sample_s1_ids,
    )

    # ---------------------------------------------------------------
    # Recall: prefix only
    # ---------------------------------------------------------------

    print(
        "\nPrefix-only recall:"
    )

    prefix_link, prefix_complete = (
        evaluate_recall(
            prefix_df,
            ground_truth,
            sample_s1_ids,
        )
    )

    print(
        f"  Link recall: "
        f"{prefix_link:.6f}"
    )

    print(
        f"  Complete entity recall: "
        f"{prefix_complete:.6f}"
    )

    print_statistics(
        prefix_df,
        sample_s1_ids,
    )

    # ---------------------------------------------------------------
    # Recall: combined
    # ---------------------------------------------------------------

    print(
        "\nCombined recall:"
    )

    combined_link, combined_complete = (
        evaluate_recall(
            combined,
            ground_truth,
            sample_s1_ids,
        )
    )

    print(
        f"  Link recall: "
        f"{combined_link:.6f}"
    )

    print(
        f"  Complete entity recall: "
        f"{combined_complete:.6f}"
    )

    print_statistics(
        combined,
        sample_s1_ids,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "PREFIX BLOCKING SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"""
Existing:
  candidates = {len(existing):,}
  link recall = {existing_link:.6f}
  complete recall = {existing_complete:.6f}

Prefix only:
  candidates = {len(prefix_df):,}
  link recall = {prefix_link:.6f}
  complete recall = {prefix_complete:.6f}

Combined:
  candidates = {len(combined):,}
  link recall = {combined_link:.6f}
  complete recall = {combined_complete:.6f}
"""
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {PREFIX_CANDIDATES}"
    )

    print(
        f"  {COMBINED_CANDIDATES}"
    )

    con.close()


if __name__ == "__main__":
    main()