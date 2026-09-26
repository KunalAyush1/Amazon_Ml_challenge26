from __future__ import annotations

import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.preprocessing.normalization import (
    normalize_dataframe,
)


TEST_DIR = PROJECT_ROOT / "data" / "test"
WORK_DIR = PROJECT_ROOT / "data" / "submission_work"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHUNK_SIZE = 100_000

# Maximum frequency for a key to participate in rare-key blocking.
MAX_CANDIDATE_KEY_FREQUENCY = 20

# Stricter frequency for the final conservative matching rule.
MAX_MATCH_KEY_FREQUENCY = 3


# ---------------------------------------------------------------------
# Step 1: Normalize test data incrementally
# ---------------------------------------------------------------------


def prepare_source(
    source_name: str,
) -> None:
    """Normalize one test source in chunks and write Parquet parts."""

    input_path = (
        TEST_DIR / f"test_{source_name}.tsv"
    )

    output_dir = (
        WORK_DIR / f"test_{source_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old parquet parts.
    for path in output_dir.glob("part_*.parquet"):
        path.unlink()

    print(
        f"\nPreparing {source_name}: "
        f"{input_path}"
    )

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    total_rows = 0
    part_number = 0

    for chunk in reader:
        normalized = normalize_dataframe(chunk)

        output_path = (
            output_dir
            / f"part_{part_number:05d}.parquet"
        )

        normalized.to_parquet(
            output_path,
            index=False,
        )

        total_rows += len(normalized)
        part_number += 1

        print(
            f"  part {part_number:04d}: "
            f"{len(normalized):,} rows"
        )

    print(
        f"Finished {source_name}: "
        f"{total_rows:,} rows"
    )


def prepare_test_data() -> None:
    """Normalize all test sources incrementally."""

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    prepare_source("source1")
    prepare_source("source2")
    prepare_source("source3")


# ---------------------------------------------------------------------
# Step 2: Build candidate set + final conservative matches
# ---------------------------------------------------------------------


def create_submission_files() -> None:
    """Create candidate_pairs.tsv and matching_results.tsv."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching_path = (
        OUTPUT_DIR / "matching_results.tsv"
    )

    candidate_path = (
        OUTPUT_DIR / "candidate_pairs.tsv"
    )

    s1_glob = str(
        WORK_DIR
        / "test_source1"
        / "part_*.parquet"
    )

    s2_glob = str(
        WORK_DIR
        / "test_source2"
        / "part_*.parquet"
    )

    s3_glob = str(
        WORK_DIR
        / "test_source3"
        / "part_*.parquet"
    )

    con = duckdb.connect()

    # Use available CPU threads for DuckDB.
    con.execute(
        "PRAGMA threads=4"
    )

    print("\nCreating normalized views...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1 AS
        SELECT
            entity_id AS source1_entity_id,
            name_compact,
            address_compact,
            country
        FROM read_parquet('{s1_glob}');
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s2 AS
        SELECT
            entity_id AS candidate_entity_id,
            name_compact,
            address_compact,
            country
        FROM read_parquet('{s2_glob}');
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s3 AS
        SELECT
            entity_id AS candidate_entity_id,
            name_compact,
            address_compact,
            country
        FROM read_parquet('{s3_glob}');
        """
    )

    # Combine S2/S3 for frequency calculations.
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW targets AS

        SELECT
            candidate_entity_id,
            name_compact,
            address_compact,
            country,
            'S2' AS source
        FROM s2

        UNION ALL

        SELECT
            candidate_entity_id,
            name_compact,
            address_compact,
            country,
            'S3' AS source
        FROM s3;
        """
    )

    # ---------------------------------------------------------------
    # Frequency tables
    # ---------------------------------------------------------------

    print("Calculating target key frequencies...")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW name_freq AS
        SELECT
            name_compact,
            COUNT(*) AS frequency
        FROM targets
        WHERE name_compact <> ''
        GROUP BY name_compact;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW address_freq AS
        SELECT
            address_compact,
            COUNT(*) AS frequency
        FROM targets
        WHERE address_compact <> ''
        GROUP BY address_compact;
        """
    )

    # ---------------------------------------------------------------
    # Candidate generation
    #
    # Candidate set contains:
    #   1. exact name + exact address
    #   2. rare exact normalized name
    #   3. rare exact normalized address
    #
    # Candidate generation does NOT decide final matches.
    # ---------------------------------------------------------------

    print("Generating candidate pairs...")

    candidate_query = f"""
        -- Strong exact name + address candidate.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id
        FROM s1
        INNER JOIN targets t
            ON s1.name_compact = t.name_compact
            AND s1.address_compact = t.address_compact
            AND s1.name_compact <> ''
            AND s1.address_compact <> ''

        UNION

        -- Rare normalized name candidate.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id
        FROM s1
        INNER JOIN targets t
            ON s1.name_compact = t.name_compact
            AND s1.name_compact <> ''
        INNER JOIN name_freq nf
            ON t.name_compact = nf.name_compact
        WHERE nf.frequency <= {MAX_CANDIDATE_KEY_FREQUENCY}

        UNION

        -- Rare normalized address candidate.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id
        FROM s1
        INNER JOIN targets t
            ON s1.address_compact = t.address_compact
            AND s1.address_compact <> ''
        INNER JOIN address_freq af
            ON t.address_compact = af.address_compact
        WHERE af.frequency <= {MAX_CANDIDATE_KEY_FREQUENCY}
    """

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW candidates AS

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM (
            {candidate_query}
        );
        """
    )

    candidate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidates;
        """
    ).fetchone()[0]

    print(
        f"Candidate pairs: "
        f"{candidate_count:,}"
    )

    # ---------------------------------------------------------------
    # candidate_pairs.tsv
    # ---------------------------------------------------------------

    print(
        "\nWriting candidate_pairs.tsv..."
    )

    con.execute(
        f"""
        COPY (

            SELECT
                s1.source1_entity_id,

                COALESCE(
                    string_agg(
                        c.candidate_entity_id,
                        ','
                        ORDER BY c.candidate_entity_id
                    ),
                    ''
                ) AS candidate_entity_ids

            FROM s1

            LEFT JOIN candidates c
                ON s1.source1_entity_id =
                   c.source1_entity_id

            GROUP BY
                s1.source1_entity_id

            ORDER BY
                s1.source1_entity_id

        )
        TO '{candidate_path}'
        (
            HEADER,
            DELIMITER '\\t'
        );
        """
    )

    # ---------------------------------------------------------------
    # Final matching rules
    #
    # This is intentionally conservative for submission #1.
    #
    # A candidate is selected when:
    #
    #   A) exact normalized name + exact normalized address
    #
    # OR
    #
    #   B) rare exact normalized name + same country
    #
    # OR
    #
    #   C) rare exact normalized address + same country
    # ---------------------------------------------------------------

    print(
        "\nGenerating final matches..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW matches AS

        -- A: exact normalized name + address.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.name_compact = t.name_compact
            AND s1.address_compact = t.address_compact
            AND s1.name_compact <> ''
            AND s1.address_compact <> ''

        WHERE EXISTS (
            SELECT 1
            FROM candidates c
            WHERE c.source1_entity_id =
                  s1.source1_entity_id
              AND c.candidate_entity_id =
                  t.candidate_entity_id
        )

        UNION

        -- B: rare exact name + same country.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.name_compact = t.name_compact
            AND s1.country = t.country
            AND s1.name_compact <> ''

        INNER JOIN name_freq nf
            ON t.name_compact = nf.name_compact

        WHERE nf.frequency <= {MAX_MATCH_KEY_FREQUENCY}

          AND EXISTS (
              SELECT 1
              FROM candidates c
              WHERE c.source1_entity_id =
                    s1.source1_entity_id
                AND c.candidate_entity_id =
                    t.candidate_entity_id
          )

        UNION

        -- C: rare exact address + same country.
        SELECT
            s1.source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.address_compact = t.address_compact
            AND s1.country = t.country
            AND s1.address_compact <> ''

        INNER JOIN address_freq af
            ON t.address_compact = af.address_compact

        WHERE af.frequency <= {MAX_MATCH_KEY_FREQUENCY}

          AND EXISTS (
              SELECT 1
              FROM candidates c
              WHERE c.source1_entity_id =
                    s1.source1_entity_id
                AND c.candidate_entity_id =
                    t.candidate_entity_id
          );
        """
    )

    match_count = con.execute(
        """
        SELECT COUNT(*)
        FROM matches;
        """
    ).fetchone()[0]

    matched_entities = con.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM matches;
        """
    ).fetchone()[0]

    total_s1 = con.execute(
        """
        SELECT COUNT(*)
        FROM s1;
        """
    ).fetchone()[0]

    print(
        f"Matched pairs: "
        f"{match_count:,}"
    )

    print(
        f"Source-1 entities with >=1 match: "
        f"{matched_entities:,}"
    )

    print(
        f"Total Source-1 entities: "
        f"{total_s1:,}"
    )

    print(
        f"Singleton/no-match entities: "
        f"{total_s1 - matched_entities:,}"
    )

    # ---------------------------------------------------------------
    # matching_results.tsv
    # ---------------------------------------------------------------

    print(
        "\nWriting matching_results.tsv..."
    )

    con.execute(
        f"""
        COPY (

            SELECT
                s1.source1_entity_id,

                COALESCE(
                    string_agg(
                        m.candidate_entity_id,
                        ','
                        ORDER BY m.candidate_entity_id
                    ),
                    ''
                ) AS matched_entity_ids

            FROM s1

            LEFT JOIN matches m
                ON s1.source1_entity_id =
                   m.source1_entity_id

            GROUP BY
                s1.source1_entity_id

            ORDER BY
                s1.source1_entity_id

        )
        TO '{matching_path}'
        (
            HEADER,
            DELIMITER '\\t'
        );
        """
    )

    con.close()

    print("\nFiles created:")
    print(
        f"  {candidate_path}"
    )
    print(
        f"  {matching_path}"
    )


def main() -> None:
    print("=" * 72)
    print("AMAZON ML CHALLENGE — FIRST SUBMISSION")
    print("=" * 72)

    prepare_test_data()
    create_submission_files()

    print("\nSubmission files generated successfully.")


if __name__ == "__main__":
    main()