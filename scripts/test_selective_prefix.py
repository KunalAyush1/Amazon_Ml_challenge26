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

EXISTING_CANDIDATES = (
    ARTIFACT_DIR
    / "submission2_candidates_2000.parquet"
)

S1_SAMPLE = 2_000

DUCKDB_THREADS = 4

# Safety guard.
MAX_COMBINED_CANDIDATES = 2_000_000


# ---------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------

CONFIGS = [
    {
        "name": "p7_a12_f50",
        "name_prefix": 7,
        "address_prefix": 12,
        "frequency_cap": 50,
    },
    {
        "name": "p7_a12_f100",
        "name_prefix": 7,
        "address_prefix": 12,
        "frequency_cap": 100,
    },
    {
        "name": "p6_a10_f50",
        "name_prefix": 6,
        "address_prefix": 10,
        "frequency_cap": 50,
    },
    {
        "name": "p6_a10_f100",
        "name_prefix": 6,
        "address_prefix": 10,
        "frequency_cap": 100,
    },
    {
        "name": "p7_a10_f100",
        "name_prefix": 7,
        "address_prefix": 10,
        "frequency_cap": 100,
    },
]


# ---------------------------------------------------------------------
# Deterministic S1 sample
# ---------------------------------------------------------------------


def get_sample_s1_ids(
    con: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Recover the exact deterministic 2,000-S1 sample."""

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
        retrieved_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    complete_recall = (
        complete_entities / eligible_entities
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


def print_stats(
    candidate_df: pd.DataFrame,
    sample_s1_ids: list[str],
) -> tuple[float, float, int]:
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

    mean_count = float(
        counts.mean()
    )

    median_count = float(
        counts.median()
    )

    p95_count = float(
        counts.quantile(0.95)
    )

    max_count = int(
        counts.max()
    )

    zero_count = int(
        (counts == 0).sum()
    )

    print(
        f"  Mean candidates/S1: "
        f"{mean_count:.3f}"
    )

    print(
        f"  Median candidates/S1: "
        f"{median_count:.3f}"
    )

    print(
        f"  P95 candidates/S1: "
        f"{p95_count:.3f}"
    )

    print(
        f"  Max candidates/S1: "
        f"{max_count:,}"
    )

    print(
        f"  Zero-candidate S1: "
        f"{zero_count:,}"
    )

    return (
        mean_count,
        p95_count,
        zero_count,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print(
        "SELECTIVE PREFIX BLOCKING — 2,000 S1"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # Check existing artifact
    # ---------------------------------------------------------------

    if not EXISTING_CANDIDATES.exists():
        raise FileNotFoundError(
            "Missing existing candidate artifact:\n"
            f"{EXISTING_CANDIDATES}"
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

    ground_truth: dict[
        str,
        set[str],
    ] = {
        str(entity_id):
            set(matches or [])
        for (
            entity_id,
            matches,
        ) in raw_ground_truth.items()
    }

    # ---------------------------------------------------------------
    # Existing candidates
    # ---------------------------------------------------------------

    print(
        "\nLoading existing candidates..."
    )

    existing = pd.read_parquet(
        EXISTING_CANDIDATES
    )

    existing = existing[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    existing[
        "source1_entity_id"
    ] = existing[
        "source1_entity_id"
    ].astype(str)

    existing[
        "candidate_entity_id"
    ] = existing[
        "candidate_entity_id"
    ].astype(str)

    print(
        f"Existing candidates: "
        f"{len(existing):,}"
    )

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
    # IMPORTANT:
    # Register existing pandas candidates so the SQL loop can
    # reference existing_candidates_view.
    # ---------------------------------------------------------------

    con.register(
        "existing_candidates_df",
        existing,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW
        existing_candidates_view AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM existing_candidates_df;
        """
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
    # Normalized Parquet views
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

    print(
        "\nCreating normalized views..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1 AS

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
        CREATE OR REPLACE TEMP VIEW targets AS

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

    results: list[dict[str, object]] = []

    # ---------------------------------------------------------------
    # Configuration loop
    # ---------------------------------------------------------------

    for config in CONFIGS:

        config_name = str(
            config["name"]
        )

        name_prefix = int(
            config["name_prefix"]
        )

        address_prefix = int(
            config["address_prefix"]
        )

        frequency_cap = int(
            config["frequency_cap"]
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"CONFIG: {config_name}"
        )

        print(
            f"  name prefix: "
            f"{name_prefix}"
        )

        print(
            f"  address prefix: "
            f"{address_prefix}"
        )

        print(
            f"  frequency cap: "
            f"{frequency_cap}"
        )

        # -----------------------------------------------------------
        # Prefix frequency tables
        # -----------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW name_prefix_freq AS

            SELECT
                LEFT(
                    name_compact,
                    {name_prefix}
                ) AS prefix,

                COUNT(*) AS frequency

            FROM targets

            WHERE name_compact <> ''

            GROUP BY prefix;
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW
            address_prefix_freq AS

            SELECT
                LEFT(
                    address_compact,
                    {address_prefix}
                ) AS prefix,

                COUNT(*) AS frequency

            FROM targets

            WHERE address_compact <> ''

            GROUP BY prefix;
            """
        )

        # -----------------------------------------------------------
        # Prefix candidates
        # -----------------------------------------------------------

        prefix_query = f"""
            -- =====================================================
            -- Name prefix candidate
            -- =====================================================

            SELECT
                s1.entity_id AS source1_entity_id,
                t.candidate_entity_id

            FROM s1

            INNER JOIN targets t

                ON s1.country <> ''
                AND s1.country =
                    t.country

                AND s1.name_compact <> ''
                AND t.name_compact <> ''

                AND LEFT(
                    s1.name_compact,
                    {name_prefix}
                )
                =
                LEFT(
                    t.name_compact,
                    {name_prefix}
                )

            INNER JOIN name_prefix_freq nf

                ON LEFT(
                    t.name_compact,
                    {name_prefix}
                )
                =
                nf.prefix

            WHERE nf.frequency <=
                {frequency_cap}


            UNION


            -- =====================================================
            -- Address prefix candidate
            -- =====================================================

            SELECT
                s1.entity_id AS source1_entity_id,
                t.candidate_entity_id

            FROM s1

            INNER JOIN targets t

                ON s1.country <> ''
                AND s1.country =
                    t.country

                AND s1.address_compact <> ''
                AND t.address_compact <> ''

                AND LEFT(
                    s1.address_compact,
                    {address_prefix}
                )
                =
                LEFT(
                    t.address_compact,
                    {address_prefix}
                )

            INNER JOIN address_prefix_freq af

                ON LEFT(
                    t.address_compact,
                    {address_prefix}
                )
                =
                af.prefix

            WHERE af.frequency <=
                {frequency_cap}
        """

        # -----------------------------------------------------------
        # Create selective-prefix candidates
        # -----------------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
            selective_prefix AS

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
            FROM selective_prefix;
            """
        ).fetchone()[0]

        print(
            f"Selective-prefix candidates: "
            f"{prefix_count:,}"
        )

        # -----------------------------------------------------------
        # Combine selective prefix with existing blockers
        # -----------------------------------------------------------

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            combined_selective AS

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM selective_prefix

            UNION

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM existing_candidates_view;
            """
        )

        combined_count = con.execute(
            """
            SELECT COUNT(*)
            FROM combined_selective;
            """
        ).fetchone()[0]

        print(
            f"Combined candidates: "
            f"{combined_count:,}"
        )

        # -----------------------------------------------------------
        # Safety guard
        # -----------------------------------------------------------

        if (
            combined_count
            > MAX_COMBINED_CANDIDATES
        ):

            print(
                "  SKIPPED:"
                " combined candidate set exceeds "
                f"{MAX_COMBINED_CANDIDATES:,}"
            )

            results.append(
                {
                    "config":
                        config_name,

                    "name_prefix":
                        name_prefix,

                    "address_prefix":
                        address_prefix,

                    "frequency_cap":
                        frequency_cap,

                    "selective_candidates":
                        int(prefix_count),

                    "combined_candidates":
                        int(combined_count),

                    "link_recall":
                        None,

                    "complete_entity_recall":
                        None,

                    "mean_candidates":
                        None,

                    "p95_candidates":
                        None,

                    "zero_candidate_s1":
                        None,
                }
            )

            continue

        # -----------------------------------------------------------
        # Load combined candidates
        # -----------------------------------------------------------

        combined_df = con.execute(
            """
            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM combined_selective;
            """
        ).fetch_df()

        combined_df[
            "source1_entity_id"
        ] = combined_df[
            "source1_entity_id"
        ].astype(str)

        combined_df[
            "candidate_entity_id"
        ] = combined_df[
            "candidate_entity_id"
        ].astype(str)

        # -----------------------------------------------------------
        # Recall
        # -----------------------------------------------------------

        link_recall, complete_recall = (
            evaluate_recall(
                combined_df,
                ground_truth,
                sample_s1_ids,
            )
        )

        print(
            f"  Link recall: "
            f"{link_recall:.6f}"
        )

        print(
            f"  Complete entity recall: "
            f"{complete_recall:.6f}"
        )

        # -----------------------------------------------------------
        # Statistics
        # -----------------------------------------------------------

        (
            mean_candidates,
            p95_candidates,
            zero_candidates,
        ) = print_stats(
            combined_df,
            sample_s1_ids,
        )

        # -----------------------------------------------------------
        # Save successful configuration
        # -----------------------------------------------------------

        output_path = (
            ARTIFACT_DIR
            / f"selective_prefix_{config_name}.parquet"
        )

        combined_df.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"  Saved:"
            f"\n    {output_path}"
        )

        results.append(
            {
                "config":
                    config_name,

                "name_prefix":
                    name_prefix,

                "address_prefix":
                    address_prefix,

                "frequency_cap":
                    frequency_cap,

                "selective_candidates":
                    int(prefix_count),

                "combined_candidates":
                    int(combined_count),

                "link_recall":
                    float(link_recall),

                "complete_entity_recall":
                    float(complete_recall),

                "mean_candidates":
                    float(mean_candidates),

                "p95_candidates":
                    float(p95_candidates),

                "zero_candidate_s1":
                    int(zero_candidates),
            }
        )

    # ---------------------------------------------------------------
    # Results summary
    # ---------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_path = (
        ARTIFACT_DIR
        / "selective_prefix_results.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SELECTIVE PREFIX RESULTS"
    )

    print(
        "=" * 72
    )

    if not result_df.empty:
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