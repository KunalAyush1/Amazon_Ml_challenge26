from __future__ import annotations

import csv
import json
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


from business_entity_resol.features.pair_features import (
    pair_features,
)
from business_entity_resol.models.lightgbm.model import (
    LightGBMMatcher,
)
from business_entity_resol.preprocessing.normalization import (
    normalize_dataframe,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_DIR = PROJECT_ROOT / "data" / "test"

WORK_DIR = (
    PROJECT_ROOT / "data" / "submission2_inference_work"
)

DB_PATH = (
    WORK_DIR / "submission2_inference.duckdb"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "submission2"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "submission2_lightgbm.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "submission2_metadata.json"
)

CHUNK_SIZE = 100_000

# Same blocking configuration that produced the
# 0.710819 held-out F0.5 result.
MAX_BLOCKING_FREQUENCY = 100

# Number of Source-1 entities scored in one model batch.
S1_BATCH_SIZE = 5_000

# DuckDB threads.
DUCKDB_THREADS = 4


# ---------------------------------------------------------------------
# Test-data preparation
# ---------------------------------------------------------------------


def prepare_test_source(
    source_name: str,
) -> None:
    """Normalize one test source incrementally to Parquet."""

    input_path = (
        TEST_DIR
        / f"test_{source_name}.tsv"
    )

    output_dir = (
        WORK_DIR
        / f"test_{source_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in output_dir.glob(
        "part_*.parquet"
    ):
        path.unlink()

    print(
        f"\nPreparing {source_name}:"
    )
    print(
        f"  input: {input_path}"
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

        normalized = normalize_dataframe(
            chunk
        )

        output_path = (
            output_dir
            / f"part_{part_number:05d}.parquet"
        )

        normalized.to_parquet(
            output_path,
            index=False,
        )

        total_rows += len(
            normalized
        )

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
    """Prepare normalized S1/S2/S3 test Parquet."""

    print(
        "\nPreparing normalized test data..."
    )

    prepare_test_source(
        "source1"
    )

    prepare_test_source(
        "source2"
    )

    prepare_test_source(
        "source3"
    )


# ---------------------------------------------------------------------
# DuckDB views
# ---------------------------------------------------------------------


def create_views(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Create normalized S1/S2/S3 DuckDB views."""

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

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s1 AS
        SELECT
            entity_id,
            business_name,
            business_address,
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
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s2 AS
        SELECT
            entity_id AS candidate_entity_id,
            'S2' AS candidate_source,
            business_name,
            business_address,
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
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW s3 AS
        SELECT
            entity_id AS candidate_entity_id,
            'S3' AS candidate_source,
            business_name,
            business_address,
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

    con.execute(
        """
        CREATE OR REPLACE VIEW targets AS

        SELECT *
        FROM s2

        UNION ALL

        SELECT *
        FROM s3;
        """
    )


# ---------------------------------------------------------------------
# Frequency tables
# ---------------------------------------------------------------------


def create_frequency_tables(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Create blocking frequency tables."""

    print(
        "\nCalculating target frequencies..."
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE name_freq AS

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
        CREATE OR REPLACE TABLE address_freq AS

        SELECT
            address_compact,
            COUNT(*) AS frequency

        FROM targets

        WHERE address_compact <> ''

        GROUP BY address_compact;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE name_token_freq AS

        SELECT
            token,
            COUNT(*) AS frequency

        FROM (
            SELECT DISTINCT
                candidate_entity_id,
                token

            FROM targets

            CROSS JOIN UNNEST(
                name_tokens
            ) AS u(token)

            WHERE token <> ''
        )

        GROUP BY token;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE address_token_freq AS

        SELECT
            token,
            COUNT(*) AS frequency

        FROM (
            SELECT DISTINCT
                candidate_entity_id,
                token

            FROM targets

            CROSS JOIN UNNEST(
                address_tokens
            ) AS u(token)

            WHERE token <> ''
        )

        GROUP BY token;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE number_freq AS

        SELECT
            number,
            COUNT(*) AS frequency

        FROM (
            SELECT DISTINCT
                candidate_entity_id,
                number

            FROM targets

            CROSS JOIN UNNEST(
                address_numbers
            ) AS u(number)

            WHERE number <> ''
        )

        GROUP BY number;
        """
    )


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------


def build_candidate_table(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Create the final candidate table.

    This is the EXACT candidate set that will be
    passed to LightGBM.
    """

    print(
        "\nGenerating FINAL candidate set..."
    )

    candidate_sql = f"""

        -- ==========================================================
        -- 1. Exact normalized name + exact normalized address
        -- ==========================================================

        SELECT
            s1.entity_id AS source1_entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.name_compact =
               t.name_compact

            AND s1.address_compact =
                t.address_compact

            AND s1.name_compact <> ''

            AND s1.address_compact <> ''


        UNION


        -- ==========================================================
        -- 2. Rare normalized name
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.name_compact =
               t.name_compact

            AND s1.name_compact <> ''

        INNER JOIN name_freq nf
            ON t.name_compact =
               nf.name_compact

        WHERE nf.frequency <=
            {MAX_BLOCKING_FREQUENCY}


        UNION


        -- ==========================================================
        -- 3. Rare normalized address
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id

        FROM s1

        INNER JOIN targets t
            ON s1.address_compact =
               t.address_compact

            AND s1.address_compact <> ''

        INNER JOIN address_freq af
            ON t.address_compact =
               af.address_compact

        WHERE af.frequency <=
            {MAX_BLOCKING_FREQUENCY}


        UNION


        -- ==========================================================
        -- 4. Rare name-token overlap
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id

        FROM s1

        CROSS JOIN UNNEST(
            s1.name_tokens
        ) AS s1_token(token)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                token

            FROM (
                SELECT
                    candidate_entity_id,
                    UNNEST(
                        name_tokens
                    ) AS token

                FROM targets
            )
        ) t

            ON s1_token.token =
               t.token

        INNER JOIN name_token_freq tf
            ON t.token =
               tf.token

        WHERE tf.frequency <=
            {MAX_BLOCKING_FREQUENCY}

            AND s1_token.token <> ''


        UNION


        -- ==========================================================
        -- 5. Rare address-token overlap
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id

        FROM s1

        CROSS JOIN UNNEST(
            s1.address_tokens
        ) AS s1_token(token)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                token

            FROM (
                SELECT
                    candidate_entity_id,
                    UNNEST(
                        address_tokens
                    ) AS token

                FROM targets
            )
        ) t

            ON s1_token.token =
               t.token

        INNER JOIN address_token_freq tf
            ON t.token =
               tf.token

        WHERE tf.frequency <=
            {MAX_BLOCKING_FREQUENCY}

            AND s1_token.token <> ''


        UNION


        -- ==========================================================
        -- 6. Shared address number
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id

        FROM s1

        CROSS JOIN UNNEST(
            s1.address_numbers
        ) AS s1_number(number)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                number

            FROM (
                SELECT
                    candidate_entity_id,
                    UNNEST(
                        address_numbers
                    ) AS number

                FROM targets
            )
        ) t

            ON s1_number.number =
               t.number

        INNER JOIN number_freq nf
            ON t.number =
               nf.number

        WHERE nf.frequency <=
            {MAX_BLOCKING_FREQUENCY}

            AND s1_number.number <> ''
    """

    con.execute(
        f"""
        CREATE OR REPLACE TABLE candidate_pairs AS

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM (
            {candidate_sql}
        );
        """
    )

    total_candidates = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs;
        """
    ).fetchone()[0]

    total_s1 = con.execute(
        """
        SELECT COUNT(*)
        FROM s1;
        """
    ).fetchone()[0]

    print(
        f"Test S1 entities: "
        f"{total_s1:,}"
    )

    print(
        f"FINAL candidate pairs: "
        f"{total_candidates:,}"
    )

    mean_candidates = con.execute(
        """
        SELECT
            AVG(candidate_count)
        FROM (
            SELECT
                source1_entity_id,
                COUNT(*) AS candidate_count
            FROM candidate_pairs
            GROUP BY source1_entity_id
        );
        """
    ).fetchone()[0]

    zero_candidates = con.execute(
        """
        SELECT COUNT(*)
        FROM s1
        LEFT JOIN candidate_pairs c
            ON s1.entity_id =
               c.source1_entity_id
        WHERE c.source1_entity_id IS NULL;
        """
    ).fetchone()[0]

    print(
        f"Mean candidates/S1: "
        f"{float(mean_candidates or 0):.3f}"
    )

    print(
        f"Zero-candidate S1: "
        f"{zero_candidates:,}"
    )


# ---------------------------------------------------------------------
# Candidate output
# ---------------------------------------------------------------------


def write_candidate_output(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> None:
    """Write candidate_pairs.tsv with exact empty-field formatting."""

    print(
        f"\nWriting candidate file:"
        f"\n  {path}"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    query = """
        SELECT
            s1.entity_id AS source1_entity_id,

            COALESCE(
                string_agg(
                    c.candidate_entity_id,
                    ','
                    ORDER BY c.candidate_entity_id
                ),
                ''
            ) AS candidate_entity_ids

        FROM s1

        LEFT JOIN candidate_pairs c
            ON s1.entity_id =
               c.source1_entity_id

        GROUP BY
            s1.entity_id

        ORDER BY
            s1.entity_id;
    """

    result = con.execute(
        query
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        while True:

            rows = result.fetchmany(
                50_000
            )

            if not rows:
                break

            writer.writerows(
                rows
            )


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------


def build_feature_dataframe(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate model features for one candidate batch."""

    rows: list[
        dict[str, object]
    ] = []

    records = candidate_df.to_dict(
        orient="records"
    )

    for row in records:

        source1 = {
            "entity_id":
                row[
                    "source1_entity_id"
                ],

            "business_name":
                row[
                    "source1_business_name"
                ],

            "business_address":
                row[
                    "source1_business_address"
                ],

            "country":
                row[
                    "source1_country"
                ],

            "name_norm":
                row[
                    "source1_name_norm"
                ],

            "name_compact":
                row[
                    "source1_name_compact"
                ],

            "address_norm":
                row[
                    "source1_address_norm"
                ],

            "address_compact":
                row[
                    "source1_address_compact"
                ],

            "address_numbers":
                row[
                    "source1_address_numbers"
                ],
        }

        candidate = {
            "entity_id":
                row[
                    "candidate_entity_id"
                ],

            "business_name":
                row[
                    "candidate_business_name"
                ],

            "business_address":
                row[
                    "candidate_business_address"
                ],

            "country":
                row[
                    "candidate_country"
                ],

            "name_norm":
                row[
                    "candidate_name_norm"
                ],

            "name_compact":
                row[
                    "candidate_name_compact"
                ],

            "address_norm":
                row[
                    "candidate_address_norm"
                ],

            "address_compact":
                row[
                    "candidate_address_compact"
                ],

            "address_numbers":
                row[
                    "candidate_address_numbers"
                ],
        }

        features = pair_features(
            source1,
            candidate,
        )

        result = {
            "source1_entity_id":
                str(
                    row[
                        "source1_entity_id"
                    ]
                ),

            "candidate_entity_id":
                str(
                    row[
                        "candidate_entity_id"
                    ]
                ),

            "candidate_is_s2":
                int(
                    str(
                        row[
                            "candidate_entity_id"
                        ]
                    ).startswith(
                        "S2-"
                    )
                ),

            "candidate_is_s3":
                int(
                    str(
                        row[
                            "candidate_entity_id"
                        ]
                    ).startswith(
                        "S3-"
                    )
                ),
        }

        result.update(
            features
        )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Create batched candidate table
# ---------------------------------------------------------------------


def create_candidate_batches(
    con: duckdb.DuckDBPyConnection,
) -> int:
    """Assign candidate pairs to Source-1 inference batches."""

    print(
        "\nCreating inference batches..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE s1_batches AS

        SELECT
            entity_id AS source1_entity_id,

            CAST(
                FLOOR(
                    (
                        ROW_NUMBER()
                        OVER (
                            ORDER BY entity_id
                        ) - 1
                    )
                    / {S1_BATCH_SIZE}
                )
                AS BIGINT
            ) AS batch_id

        FROM s1;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_pairs_batched AS

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            b.batch_id

        FROM candidate_pairs c

        INNER JOIN s1_batches b
            ON c.source1_entity_id =
               b.source1_entity_id;
        """
    )

    batch_count = con.execute(
        """
        SELECT
            COALESCE(
                MAX(batch_id),
                -1
            ) + 1
        FROM s1_batches;
        """
    ).fetchone()[0]

    print(
        f"Inference batches: "
        f"{batch_count:,}"
    )

    return int(
        batch_count
    )


# ---------------------------------------------------------------------
# Fetch one candidate batch
# ---------------------------------------------------------------------


def fetch_candidate_batch(
    con: duckdb.DuckDBPyConnection,
    batch_id: int,
) -> pd.DataFrame:
    """Fetch complete candidate records for one S1 batch."""

    query = """
        SELECT
            cp.source1_entity_id,
            cp.candidate_entity_id,

            s1.business_name
                AS source1_business_name,

            s1.business_address
                AS source1_business_address,

            s1.country
                AS source1_country,

            s1.name_norm
                AS source1_name_norm,

            s1.name_compact
                AS source1_name_compact,

            s1.address_norm
                AS source1_address_norm,

            s1.address_compact
                AS source1_address_compact,

            s1.address_numbers
                AS source1_address_numbers,

            t.business_name
                AS candidate_business_name,

            t.business_address
                AS candidate_business_address,

            t.country
                AS candidate_country,

            t.name_norm
                AS candidate_name_norm,

            t.name_compact
                AS candidate_name_compact,

            t.address_norm
                AS candidate_address_norm,

            t.address_compact
                AS candidate_address_compact,

            t.address_numbers
                AS candidate_address_numbers,

            t.candidate_source

        FROM candidate_pairs_batched cp

        INNER JOIN s1
            ON cp.source1_entity_id =
               s1.entity_id

        INNER JOIN targets t
            ON cp.candidate_entity_id =
               t.candidate_entity_id

        WHERE cp.batch_id = ?

        ORDER BY
            cp.source1_entity_id,
            cp.candidate_entity_id;
    """

    return con.execute(
        query,
        [batch_id],
    ).fetch_df()


# ---------------------------------------------------------------------
# Score one batch
# ---------------------------------------------------------------------


def score_batch(
    model: LightGBMMatcher,
    candidate_df: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
) -> dict[
    str,
    dict[str, list[str]],
]:
    """
    Score one candidate batch.

    Returns
    -------
    mapping:
        {
            source1_id: {
                "candidates": [...],
                "matches": [...],
            }
        }
    """

    if candidate_df.empty:
        return {}

    feature_df = build_feature_dataframe(
        candidate_df
    )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Model features missing from inference dataframe: "
            f"{missing_features}"
        )

    X = feature_df[
        feature_names
    ].copy()

    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "Non-numeric inference features: "
            f"{non_numeric}"
        )

    probabilities = model.predict_proba(
        X
    )

    feature_df[
        "match_probability"
    ] = probabilities

    results: dict[
        str,
        dict[str, list[str]],
    ] = {}

    for source1_id, group in (
        feature_df
        .groupby(
            "source1_entity_id",
            sort=False,
        )
    ):

        candidate_ids = (
            group[
                "candidate_entity_id"
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        matches = (
            group[
                group[
                    "match_probability"
                ]
                >= threshold
            ][
                "candidate_entity_id"
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        matches.sort()

        results[
            str(source1_id)
        ] = {
            "candidates":
                sorted(candidate_ids),

            "matches":
                matches,
        }

    return results


# ---------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------


def write_matching_output(
    output_path: Path,
    all_results: dict[
        str,
        dict[str, list[str]],
    ],
) -> None:
    """Write matching_results.tsv."""

    print(
        f"\nWriting matching results:"
        f"\n  {output_path}"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        for source1_id in sorted(
            all_results
        ):

            matches = all_results[
                source1_id
            ][
                "matches"
            ]

            writer.writerow(
                [
                    source1_id,
                    ",".join(matches),
                ]
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print(
        "SUBMISSION #2 — FULL TEST INFERENCE"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Load trained model and metadata.
    # ---------------------------------------------------------------

    print(
        "\nLoading Submission #2 model..."
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    model = LightGBMMatcher.load(
        MODEL_PATH
    )

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    threshold = float(
        metadata[
            "threshold"
        ]
    )

    feature_names = list(
        metadata[
            "features"
        ]
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        f"Features: "
        f"{len(feature_names)}"
    )

    print(
        f"Expected held-out F0.5: "
        f"{metadata.get('held_out_f05')}"
    )

    # ---------------------------------------------------------------
    # 2. Prepare normalized test data.
    # ---------------------------------------------------------------

    if WORK_DIR.exists():
        print(
            "\nRemoving previous inference work..."
        )

        shutil.rmtree(
            WORK_DIR
        )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_test_data()

    # ---------------------------------------------------------------
    # 3. Open persistent DuckDB database.
    # ---------------------------------------------------------------

    con = duckdb.connect(
        str(DB_PATH)
    )

    con.execute(
        f"PRAGMA threads={DUCKDB_THREADS}"
    )

    # ---------------------------------------------------------------
    # 4. Create views/frequency tables.
    # ---------------------------------------------------------------

    print(
        "\nCreating DuckDB views..."
    )

    create_views(
        con
    )

    create_frequency_tables(
        con
    )

    # ---------------------------------------------------------------
    # 5. Generate final candidate set.
    # ---------------------------------------------------------------

    build_candidate_table(
        con
    )

    # ---------------------------------------------------------------
    # 6. Write candidate_pairs.tsv FIRST.
    # ---------------------------------------------------------------

    output_dir = OUTPUT_DIR

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_output = (
        output_dir
        / "candidate_pairs.tsv"
    )

    matching_output = (
        output_dir
        / "matching_results.tsv"
    )

    write_candidate_output(
        con,
        candidate_output,
    )

    # ---------------------------------------------------------------
    # 7. Create inference batches.
    # ---------------------------------------------------------------

    batch_count = (
        create_candidate_batches(
            con
        )
    )

    # ---------------------------------------------------------------
    # 8. Score every candidate.
    # ---------------------------------------------------------------

    print(
        "\nScoring FINAL candidates..."
    )

    all_results: dict[
        str,
        dict[str, list[str]],
    ] = {}

    total_scored = 0
    total_matches = 0

    for batch_id in range(
        batch_count
    ):

        candidate_df = (
            fetch_candidate_batch(
                con,
                batch_id,
            )
        )

        batch_results = score_batch(
            model=model,
            candidate_df=candidate_df,
            feature_names=feature_names,
            threshold=threshold,
        )

        for source1_id, result in (
            batch_results.items()
        ):

            all_results[
                source1_id
            ] = result

            total_scored += len(
                result[
                    "candidates"
                ]
            )

            total_matches += len(
                result[
                    "matches"
                ]
            )

        print(
            f"  batch "
            f"{batch_id + 1:,}/"
            f"{batch_count:,} | "
            f"candidates scored: "
            f"{total_scored:,} | "
            f"matches selected: "
            f"{total_matches:,}"
        )

    # ---------------------------------------------------------------
    # 9. Ensure EVERY test S1 gets an output row.
    # ---------------------------------------------------------------

    test_s1_ids = [
        str(row[0])
        for row in con.execute(
            """
            SELECT entity_id
            FROM s1
            ORDER BY entity_id;
            """
        ).fetchall()
    ]

    missing_s1 = [
        entity_id
        for entity_id in test_s1_ids
        if entity_id not in all_results
    ]

    for entity_id in missing_s1:
        all_results[
            entity_id
        ] = {
            "candidates": [],
            "matches": [],
        }

    print(
        f"\nTest S1 entities: "
        f"{len(test_s1_ids):,}"
    )

    print(
        f"Output S1 entities: "
        f"{len(all_results):,}"
    )

    print(
        f"S1 entities with no candidates: "
        f"{len(missing_s1):,}"
    )

    # ---------------------------------------------------------------
    # 10. Final consistency check:
    # matches MUST be subset of candidates.
    # ---------------------------------------------------------------

    for source1_id, result in (
        all_results.items()
    ):

        candidate_set = set(
            result[
                "candidates"
            ]
        )

        match_set = set(
            result[
                "matches"
            ]
        )

        invalid_matches = (
            match_set
            - candidate_set
        )

        if invalid_matches:
            raise RuntimeError(
                "Submission consistency failure: "
                f"{source1_id} contains matches "
                f"not present in its candidates: "
                f"{sorted(invalid_matches)[:10]}"
            )

    # ---------------------------------------------------------------
    # 11. Write matching_results.tsv.
    # ---------------------------------------------------------------

    write_matching_output(
        matching_output,
        all_results,
    )

    # ---------------------------------------------------------------
    # 12. Final counts.
    # ---------------------------------------------------------------

    non_empty_matches = sum(
        bool(
            result["matches"]
        )
        for result in all_results.values()
    )

    empty_matches = (
        len(all_results)
        - non_empty_matches
    )

    print(
        "\nFinal output statistics:"
    )

    print(
        f"  S1 rows: "
        f"{len(all_results):,}"
    )

    print(
        f"  Non-empty matches: "
        f"{non_empty_matches:,}"
    )

    print(
        f"  Empty matches: "
        f"{empty_matches:,}"
    )

    print(
        f"  Candidate pairs scored: "
        f"{total_scored:,}"
    )

    print(
        f"  Matches selected: "
        f"{total_matches:,}"
    )

    con.close()

    print(
        "\n" + "=" * 72
    )

    print(
        "SUBMISSION #2 INFERENCE COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nCandidate file:"
        f"\n  {candidate_output}"
    )

    print(
        f"\nMatching file:"
        f"\n  {matching_output}"
    )


if __name__ == "__main__":
    main()