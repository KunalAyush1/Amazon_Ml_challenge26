from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

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

COMBINED_CANDIDATES = (
    ARTIFACT_DIR
    / "submission2_combined_candidates_2000.parquet"
)

PRUNED_OUTPUT_TEMPLATE = (
    ARTIFACT_DIR
    / "submission2_prefix_pruned_k{K}.parquet"
)

S1_SAMPLE = 2_000

# Experiments to test.
K_VALUES = [
    50,
    100,
    200,
    500,
]

DUCKDB_THREADS = 4


# ---------------------------------------------------------------------
# Deterministic S1 sample
# ---------------------------------------------------------------------


def get_sample_s1_ids(
    con: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Recover the exact 2,000-S1 sample used in training."""

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
# Recall evaluation
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

        candidate_set = (
            candidates_by_s1.get(
                source1_id,
                set(),
            )
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


def candidate_statistics(
    candidate_df: pd.DataFrame,
    sample_s1_ids: list[str],
) -> tuple[float, float, int]:
    """Calculate candidate-count statistics."""

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

    return (
        float(counts.mean()),
        float(counts.quantile(0.95)),
        int((counts == 0).sum()),
    )


# ---------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------


def create_scored_candidates(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Create a scored version of the combined candidate set.

    This is deliberately cheap. It is a PRE-RANKER, not the final
    LightGBM model.

    Evidence used:

        exact normalized name
        exact normalized address
        same country
        same name prefix
        same address prefix
        shared name token
        shared address token
        shared address number
        name/address length similarity
    """

    combined_glob = str(
        COMBINED_CANDIDATES
    )

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

    print(
        "\nCreating normalized views..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1 AS

        SELECT
            entity_id,
            country,
            name_norm,
            name_compact,
            name_tokens,
            address_norm,
            address_compact,
            address_tokens,
            address_numbers

        FROM read_parquet(
            '{s1_glob}'
        )

        ORDER BY hash(entity_id)

        LIMIT {S1_SAMPLE};
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW targets AS

        SELECT
            entity_id AS candidate_entity_id,
            country,
            name_norm,
            name_compact,
            name_tokens,
            address_norm,
            address_compact,
            address_tokens,
            address_numbers

        FROM read_parquet(
            '{s2_glob}'
        )

        UNION ALL

        SELECT
            entity_id AS candidate_entity_id,
            country,
            name_norm,
            name_compact,
            name_tokens,
            address_norm,
            address_compact,
            address_tokens,
            address_numbers

        FROM read_parquet(
            '{s3_glob}'
        );
        """
    )

    print(
        "\nJoining combined candidates..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE scored_candidates AS

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,

            (
                -- Exact name is extremely strong.
                CASE
                    WHEN
                        s1.name_compact <> ''
                        AND
                        s1.name_compact =
                        t.name_compact
                    THEN 1000
                    ELSE 0
                END

                +

                -- Exact address is extremely strong.
                CASE
                    WHEN
                        s1.address_compact <> ''
                        AND
                        s1.address_compact =
                        t.address_compact
                    THEN 1000
                    ELSE 0
                END

                +

                -- Country agreement.
                CASE
                    WHEN
                        s1.country <> ''
                        AND
                        s1.country =
                        t.country
                    THEN 100
                    ELSE 0
                END

                +

                -- Name prefix agreement.
                CASE
                    WHEN
                        s1.name_compact <> ''
                        AND
                        t.name_compact <> ''
                        AND
                        LEFT(
                            s1.name_compact,
                            5
                        )
                        =
                        LEFT(
                            t.name_compact,
                            5
                        )
                    THEN 40
                    ELSE 0
                END

                +

                -- Address prefix agreement.
                CASE
                    WHEN
                        s1.address_compact <> ''
                        AND
                        t.address_compact <> ''
                        AND
                        LEFT(
                            s1.address_compact,
                            8
                        )
                        =
                        LEFT(
                            t.address_compact,
                            8
                        )
                    THEN 40
                    ELSE 0
                END

                +

                -- Shared name token.
                CASE
                    WHEN
                        list_has_any(
                            s1.name_tokens,
                            t.name_tokens
                        )
                    THEN 30
                    ELSE 0
                END

                +

                -- Shared address token.
                CASE
                    WHEN
                        list_has_any(
                            s1.address_tokens,
                            t.address_tokens
                        )
                    THEN 30
                    ELSE 0
                END

                +

                -- Shared address number.
                CASE
                    WHEN
                        list_has_any(
                            s1.address_numbers,
                            t.address_numbers
                        )
                    THEN 50
                    ELSE 0
                END

                +

                -- Prefer similar name lengths.
                CASE
                    WHEN
                        ABS(
                            LENGTH(
                                s1.name_compact
                            )
                            -
                            LENGTH(
                                t.name_compact
                            )
                        ) <= 2
                    THEN 10
                    WHEN
                        ABS(
                            LENGTH(
                                s1.name_compact
                            )
                            -
                            LENGTH(
                                t.name_compact
                            )
                        ) <= 5
                    THEN 5
                    ELSE 0
                END

                +

                -- Prefer similar address lengths.
                CASE
                    WHEN
                        ABS(
                            LENGTH(
                                s1.address_compact
                            )
                            -
                            LENGTH(
                                t.address_compact
                            )
                        ) <= 5
                    THEN 10
                    WHEN
                        ABS(
                            LENGTH(
                                s1.address_compact
                            )
                            -
                            LENGTH(
                                t.address_compact
                            )
                        ) <= 12
                    THEN 5
                    ELSE 0
                END
            ) AS heuristic_score

        FROM read_parquet(
            '{combined_glob}'
        ) c

        INNER JOIN s1
            ON c.source1_entity_id =
               s1.entity_id

        INNER JOIN targets t
            ON c.candidate_entity_id =
               t.candidate_entity_id;
        """
    )

    total = con.execute(
        """
        SELECT COUNT(*)
        FROM scored_candidates;
        """
    ).fetchone()[0]

    print(
        f"Scored candidates: "
        f"{total:,}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print(
        "PREFIX CANDIDATE PRUNING EXPERIMENT"
    )
    print("=" * 72)

    if not COMBINED_CANDIDATES.exists():
        raise FileNotFoundError(
            "Missing combined candidate artifact:\n"
            f"{COMBINED_CANDIDATES}"
        )

    # ---------------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # DuckDB
    # ---------------------------------------------------------------

    con = duckdb.connect()

    con.execute(
        f"PRAGMA threads={DUCKDB_THREADS}"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    # ---------------------------------------------------------------
    # Sample IDs
    # ---------------------------------------------------------------

    sample_s1_ids = get_sample_s1_ids(
        con
    )

    print(
        f"Sample S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    # ---------------------------------------------------------------
    # Score combined candidates
    # ---------------------------------------------------------------

    create_scored_candidates(
        con
    )

    # ---------------------------------------------------------------
    # Run K experiments
    # ---------------------------------------------------------------

    results = []

    for k in K_VALUES:

        print(
            "\n" + "-" * 64
        )

        print(
            f"Testing K = {k}"
        )

        # -----------------------------------------------------------
        # Rank and prune
        # -----------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TABLE pruned_candidates AS

            SELECT
                source1_entity_id,
                candidate_entity_id,
                heuristic_score

            FROM (

                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    heuristic_score,

                    ROW_NUMBER()
                    OVER (
                        PARTITION BY
                            source1_entity_id

                        ORDER BY
                            heuristic_score DESC,
                            candidate_entity_id
                    ) AS rank

                FROM scored_candidates
            )

            WHERE rank <= {k};
            """
        )

        row_count = con.execute(
            """
            SELECT COUNT(*)
            FROM pruned_candidates;
            """
        ).fetchone()[0]

        print(
            f"Candidate rows: "
            f"{row_count:,}"
        )

        # -----------------------------------------------------------
        # Load pruned candidates for recall evaluation
        # -----------------------------------------------------------

        pruned_df = con.execute(
            """
            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM pruned_candidates;
            """
        ).fetch_df()

        link_recall, complete_recall = (
            evaluate_recall(
                pruned_df,
                ground_truth,
                sample_s1_ids,
            )
        )

        mean_candidates, p95, zero = (
            candidate_statistics(
                pruned_df,
                sample_s1_ids,
            )
        )

        print(
            f"Link recall: "
            f"{link_recall:.6f}"
        )

        print(
            f"Complete entity recall: "
            f"{complete_recall:.6f}"
        )

        print(
            f"Mean candidates/S1: "
            f"{mean_candidates:.3f}"
        )

        print(
            f"P95 candidates/S1: "
            f"{p95:.3f}"
        )

        print(
            f"Zero-candidate S1: "
            f"{zero:,}"
        )

        # -----------------------------------------------------------
        # Save this K
        # -----------------------------------------------------------

        output_path = Path(
            str(
                PRUNED_OUTPUT_TEMPLATE
            ).replace(
                "{K}",
                str(k),
            )
        )

        pruned_df.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"Saved:"
            f"\n  {output_path}"
        )

        results.append(
            {
                "K":
                    k,

                "candidate_rows":
                    row_count,

                "link_recall":
                    link_recall,

                "complete_entity_recall":
                    complete_recall,

                "mean_candidates":
                    mean_candidates,

                "p95_candidates":
                    p95,

                "zero_candidate_s1":
                    zero,
            }
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_path = (
        ARTIFACT_DIR
        / "submission2_prefix_pruning_results.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "PREFIX PRUNING RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results:"
        f"\n  {result_path}"
    )

    con.close()


if __name__ == "__main__":
    main()