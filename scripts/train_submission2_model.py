from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.evaluation.entity_f05 import (
    macro_f05,
)
from business_entity_resol.evaluation.threshold_search import (
    search_best_threshold,
)
from business_entity_resol.features.pair_features import (
    pair_features,
)
from business_entity_resol.io.ground_truth import (
    read_ground_truth,
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

S1_SAMPLE = 2_000

CHUNK_SIZE = 100_000

# Frequency cap for exact/token/number blocking.
MAX_BLOCKING_FREQUENCY = 100

RANDOM_SEED = 42

# Safety guard for the Mac experiment.
MAX_EXPERIMENT_CANDIDATES = 2_000_000


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def normalize_source_to_parquet(
    source_name: str,
) -> None:
    """Normalize one training source incrementally."""

    input_path = (
        TRAIN_DIR
        / f"train_{source_name}.tsv"
    )

    output_dir = (
        WORK_DIR
        / f"train_{source_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove previous normalized chunks.
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


def prepare_training_data() -> None:
    """Normalize train S1/S2/S3 into Parquet."""

    if WORK_DIR.exists():
        shutil.rmtree(
            WORK_DIR
        )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalize_source_to_parquet(
        "source1"
    )

    normalize_source_to_parquet(
        "source2"
    )

    normalize_source_to_parquet(
        "source3"
    )


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------


def build_candidate_dataframe(
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate candidates for a deterministic 2,000-S1 sample
    against the full Source-2 + Source-3 training population.

    Blocking rules:

        1. exact normalized name + exact normalized address
        2. rare normalized name
        3. rare normalized address
        4. rare name-token overlap
        5. rare address-token overlap
        6. shared address number

    Returns
    -------
    candidate_df:
        Candidate pairs with fields needed for feature generation.

    sample_s1_ids:
        All sampled Source-1 IDs, including entities with
        zero candidates.
    """

    con = duckdb.connect()

    con.execute(
        "PRAGMA threads=4"
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

    # ---------------------------------------------------------------
    # Source 1
    # ---------------------------------------------------------------

    print(
        f"\nSelecting deterministic "
        f"S1 sample of {S1_SAMPLE:,}..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1_full AS
        SELECT *
        FROM read_parquet(
            '{s1_glob}'
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1 AS
        SELECT *
        FROM s1_full
        ORDER BY hash(entity_id)
        LIMIT {S1_SAMPLE};
        """
    )

    sample_s1_ids = [
        str(row[0])
        for row in con.execute(
            """
            SELECT entity_id
            FROM s1
            ORDER BY entity_id;
            """
        ).fetchall()
    ]

    print(
        f"Sampled S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    # ---------------------------------------------------------------
    # Source 2
    # ---------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s2 AS
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

    # ---------------------------------------------------------------
    # Source 3
    # ---------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s3 AS
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

    # ---------------------------------------------------------------
    # Target union
    # ---------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW targets AS

        SELECT *
        FROM s2

        UNION ALL

        SELECT *
        FROM s3;
        """
    )

    # ---------------------------------------------------------------
    # Exact-key frequencies
    # ---------------------------------------------------------------

    print(
        "\nCalculating exact-key frequencies..."
    )

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
    # Token frequencies
    # ---------------------------------------------------------------

    print(
        "\nCalculating token frequencies..."
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW name_token_freq AS

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
        CREATE OR REPLACE TEMP VIEW address_token_freq AS

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

    # ---------------------------------------------------------------
    # Number frequencies
    # ---------------------------------------------------------------

    print(
        "\nCalculating address-number frequencies..."
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW number_freq AS

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

    # ---------------------------------------------------------------
    # Candidate query
    # ---------------------------------------------------------------

    print(
        "\nGenerating candidates against "
        "the FULL S2/S3 population..."
    )

    candidate_sql = f"""
        -- ==========================================================
        -- 1. Exact normalized name + exact normalized address
        -- ==========================================================

        SELECT
            s1.entity_id AS source1_entity_id,
            t.candidate_entity_id,
            t.candidate_source,

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
                AS candidate_address_numbers

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
        -- 2. Rare exact normalized name
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id,
            t.candidate_source,

            s1.business_name,
            s1.business_address,
            s1.country,
            s1.name_norm,
            s1.name_compact,
            s1.address_norm,
            s1.address_compact,
            s1.address_numbers,

            t.business_name,
            t.business_address,
            t.country,
            t.name_norm,
            t.name_compact,
            t.address_norm,
            t.address_compact,
            t.address_numbers

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
        -- 3. Rare exact normalized address
        -- ==========================================================

        SELECT
            s1.entity_id,
            t.candidate_entity_id,
            t.candidate_source,

            s1.business_name,
            s1.business_address,
            s1.country,
            s1.name_norm,
            s1.name_compact,
            s1.address_norm,
            s1.address_compact,
            s1.address_numbers,

            t.business_name,
            t.business_address,
            t.country,
            t.name_norm,
            t.name_compact,
            t.address_norm,
            t.address_compact,
            t.address_numbers

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
            t.candidate_entity_id,
            t.candidate_source,

            s1.business_name,
            s1.business_address,
            s1.country,
            s1.name_norm,
            s1.name_compact,
            s1.address_norm,
            s1.address_compact,
            s1.address_numbers,

            t.business_name,
            t.business_address,
            t.country,
            t.name_norm,
            t.name_compact,
            t.address_norm,
            t.address_compact,
            t.address_numbers

        FROM s1

        CROSS JOIN UNNEST(
            s1.name_tokens
        ) AS s1_token(token)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                candidate_source,
                business_name,
                business_address,
                country,
                name_norm,
                name_compact,
                address_norm,
                address_compact,
                address_numbers,
                token

            FROM (
                SELECT
                    *,
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
            t.candidate_entity_id,
            t.candidate_source,

            s1.business_name,
            s1.business_address,
            s1.country,
            s1.name_norm,
            s1.name_compact,
            s1.address_norm,
            s1.address_compact,
            s1.address_numbers,

            t.business_name,
            t.business_address,
            t.country,
            t.name_norm,
            t.name_compact,
            t.address_norm,
            t.address_compact,
            t.address_numbers

        FROM s1

        CROSS JOIN UNNEST(
            s1.address_tokens
        ) AS s1_token(token)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                candidate_source,
                business_name,
                business_address,
                country,
                name_norm,
                name_compact,
                address_norm,
                address_compact,
                address_numbers,
                token

            FROM (
                SELECT
                    *,
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
            t.candidate_entity_id,
            t.candidate_source,

            s1.business_name,
            s1.business_address,
            s1.country,
            s1.name_norm,
            s1.name_compact,
            s1.address_norm,
            s1.address_compact,
            s1.address_numbers,

            t.business_name,
            t.business_address,
            t.country,
            t.name_norm,
            t.name_compact,
            t.address_norm,
            t.address_compact,
            t.address_numbers

        FROM s1

        CROSS JOIN UNNEST(
            s1.address_numbers
        ) AS s1_number(number)

        INNER JOIN (
            SELECT
                candidate_entity_id,
                candidate_source,
                business_name,
                business_address,
                country,
                name_norm,
                name_compact,
                address_norm,
                address_compact,
                address_numbers,
                number

            FROM (
                SELECT
                    *,
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

    query = f"""
        SELECT DISTINCT
            *
        FROM (
            {candidate_sql}
        )
    """

    # ---------------------------------------------------------------
    # Candidate count
    # ---------------------------------------------------------------

    candidate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            {query}
        )
        """
    ).fetchone()[0]

    print(
        f"\nCandidate pairs: "
        f"{candidate_count:,}"
    )

    if (
        candidate_count
        > MAX_EXPERIMENT_CANDIDATES
    ):
        con.close()

        raise RuntimeError(
            "\nCandidate set is too large "
            "for the current experiment.\n"
            f"Found: {candidate_count:,}\n"
            f"Limit: {MAX_EXPERIMENT_CANDIDATES:,}\n"
        )

    candidate_df = con.execute(
        query
    ).fetch_df()

    con.close()

    return (
        candidate_df,
        sample_s1_ids,
    )


# ---------------------------------------------------------------------
# Candidate recall
# ---------------------------------------------------------------------


def evaluate_candidate_recall(
    candidate_df: pd.DataFrame,
    sampled_ground_truth: dict[str, set[str]],
) -> tuple[float, float]:
    """Calculate link recall and complete entity recall."""

    candidates_by_s1: dict[
        str,
        set[str],
    ] = {}

    for row in candidate_df.itertuples(
        index=False
    ):
        candidates_by_s1.setdefault(
            str(
                row.source1_entity_id
            ),
            set(),
        ).add(
            str(
                row.candidate_entity_id
            )
        )

    total_true_links = 0
    retrieved_true_links = 0

    eligible_entities = 0
    complete_entities = 0

    for source1_id, true_ids in (
        sampled_ground_truth.items()
    ):
        true_set = set(
            true_ids or set()
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


def print_candidate_statistics(
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
        "\nCandidate-count statistics:"
    )

    print(
        f"  Mean: "
        f"{counts.mean():.3f}"
    )

    print(
        f"  Median: "
        f"{counts.median():.3f}"
    )

    print(
        f"  P95: "
        f"{counts.quantile(0.95):.3f}"
    )

    print(
        f"  Max: "
        f"{counts.max():,}"
    )

    print(
        f"  Zero-candidate S1: "
        f"{(counts == 0).sum():,}"
    )


# ---------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------


def build_features(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate numeric pairwise features."""

    print(
        "\nBuilding pair features..."
    )

    rows: list[
        dict[str, object]
    ] = []

    records = candidate_df.to_dict(
        orient="records"
    )

    total = len(
        records
    )

    for index, row in enumerate(
        records,
        start=1,
    ):

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

        feature_row: dict[
            str,
            object,
        ] = {
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

            "candidate_source":
                str(
                    row[
                        "candidate_source"
                    ]
                ),

            "candidate_is_s2":
                int(
                    row[
                        "candidate_source"
                    ] == "S2"
                ),

            "candidate_is_s3":
                int(
                    row[
                        "candidate_source"
                    ] == "S3"
                ),
        }

        feature_row.update(
            features
        )

        rows.append(
            feature_row
        )

        if index % 50_000 == 0:
            print(
                f"  feature rows: "
                f"{index:,}/{total:,}"
            )

    print(
        f"Completed "
        f"{total:,} feature rows."
    )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------


def select_model_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Select model features.

    Current configuration:

        address = True
        lexical = True
        blocker = False

    The candidate experiment does not preserve blocker provenance,
    so blocker_* features are excluded.
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
        and not column.startswith(
            "blocker_"
        )
    ]


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------


def add_labels(
    feature_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
) -> pd.DataFrame:
    """Add binary match labels."""

    feature_df = feature_df.copy()

    feature_df["label"] = [
        int(
            str(candidate_id)
            in ground_truth.get(
                str(source1_id),
                set(),
            )
        )
        for (
            source1_id,
            candidate_id,
        ) in zip(
            feature_df[
                "source1_entity_id"
            ],
            feature_df[
                "candidate_entity_id"
            ],
        )
    ]

    return feature_df


# ---------------------------------------------------------------------
# Three-way group split
# ---------------------------------------------------------------------


def split_s1_entities(
    s1_ids: list[str],
) -> tuple[
    set[str],
    set[str],
    set[str],
]:
    """
    Split Source-1 entities into:

        70% train
        15% threshold tuning
        15% final evaluation
    """

    groups = pd.DataFrame(
        {
            "source1_entity_id":
                list(s1_ids)
        }
    )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    train_idx, temp_idx = next(
        first_split.split(
            groups,
            groups=groups[
                "source1_entity_id"
            ],
        )
    )

    train_ids = set(
        groups.iloc[
            train_idx
        ][
            "source1_entity_id"
        ].astype(str)
    )

    temp = groups.iloc[
        temp_idx
    ].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    tune_idx, eval_idx = next(
        second_split.split(
            temp,
            groups=temp[
                "source1_entity_id"
            ],
        )
    )

    tune_ids = set(
        temp.iloc[
            tune_idx
        ][
            "source1_entity_id"
        ].astype(str)
    )

    eval_ids = set(
        temp.iloc[
            eval_idx
        ][
            "source1_entity_id"
        ].astype(str)
    )

    # Safety checks.
    assert train_ids.isdisjoint(
        tune_ids
    )

    assert train_ids.isdisjoint(
        eval_ids
    )

    assert tune_ids.isdisjoint(
        eval_ids
    )

    assert (
        len(
            train_ids
            | tune_ids
            | eval_ids
        )
        == len(s1_ids)
    )

    return (
        train_ids,
        tune_ids,
        eval_ids,
    )


# ---------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------


def build_score_mapping(
    pair_df: pd.DataFrame,
    probabilities: object,
) -> dict[
    str,
    dict[str, float],
]:
    """Convert probabilities into nested entity/candidate mappings."""

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for row, probability in zip(
        pair_df.itertuples(
            index=False
        ),
        probabilities,
    ):

        source1_id = str(
            row.source1_entity_id
        )

        candidate_id = str(
            row.candidate_entity_id
        )

        result.setdefault(
            source1_id,
            {},
        )[candidate_id] = float(
            probability
        )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print(
        "SUBMISSION #2 — HIGH-RECALL TRAINING"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Prepare normalized training data.
    # ---------------------------------------------------------------

    prepare_training_data()

    # ---------------------------------------------------------------
    # 2. Load ground truth.
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
    # 3. Generate candidate pairs.
    # ---------------------------------------------------------------

    (
        candidate_df,
        sample_s1_ids,
    ) = build_candidate_dataframe()

    # ---------------------------------------------------------------
    # SAVE CANDIDATES FOR FUTURE PRUNING EXPERIMENTS
    # ---------------------------------------------------------------

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_artifact = (
        ARTIFACT_DIR
        / "submission2_candidates_2000.parquet"
    )

    print(
        f"\nSaving candidate dataframe:"
        f"\n  {candidate_artifact}"
    )

    candidate_df.to_parquet(
        candidate_artifact,
        index=False,
    )

    print(
        f"Saved "
        f"{len(candidate_df):,} "
        "candidate rows."
    )

    # ---------------------------------------------------------------
    # 4. Sampled ground truth.
    # ---------------------------------------------------------------

    sampled_ground_truth = {
        entity_id:
            ground_truth.get(
                entity_id,
                set(),
            )
        for entity_id
        in sample_s1_ids
    }

    # ---------------------------------------------------------------
    # 5. Candidate recall.
    # ---------------------------------------------------------------

    (
        link_recall,
        complete_recall,
    ) = evaluate_candidate_recall(
        candidate_df,
        sampled_ground_truth,
    )

    print(
        "\nCandidate recall:"
    )

    print(
        f"  Link recall: "
        f"{link_recall:.6f}"
    )

    print(
        f"  Complete entity recall: "
        f"{complete_recall:.6f}"
    )

    print_candidate_statistics(
        candidate_df,
        sample_s1_ids,
    )

    # ---------------------------------------------------------------
    # 6. Build pair features.
    # ---------------------------------------------------------------

    feature_df = build_features(
        candidate_df
    )

    # ---------------------------------------------------------------
    # Save features for pruning experiments.
    # ---------------------------------------------------------------

    feature_artifact = (
        ARTIFACT_DIR
        / "submission2_features_2000.parquet"
    )

    print(
        f"\nSaving feature dataframe:"
        f"\n  {feature_artifact}"
    )

    feature_df.to_parquet(
        feature_artifact,
        index=False,
    )

    print(
        f"Saved "
        f"{len(feature_df):,} "
        "feature rows."
    )

    # ---------------------------------------------------------------
    # 7. Add labels.
    # ---------------------------------------------------------------

    feature_df = add_labels(
        feature_df,
        ground_truth,
    )

    print(
        "\nLabel distribution:"
    )

    print(
        feature_df[
            "label"
        ].value_counts()
    )

    # ---------------------------------------------------------------
    # 8. Split S1 entities.
    # ---------------------------------------------------------------

    (
        train_ids,
        tune_ids,
        eval_ids,
    ) = split_s1_entities(
        sample_s1_ids
    )

    print(
        "\nS1 entity split:"
    )

    print(
        f"  Train: "
        f"{len(train_ids):,}"
    )

    print(
        f"  Tune:  "
        f"{len(tune_ids):,}"
    )

    print(
        f"  Eval:  "
        f"{len(eval_ids):,}"
    )

    # ---------------------------------------------------------------
    # 9. Pair-level split.
    # ---------------------------------------------------------------

    train_df = feature_df[
        feature_df[
            "source1_entity_id"
        ].isin(train_ids)
    ].copy()

    tune_df = feature_df[
        feature_df[
            "source1_entity_id"
        ].isin(tune_ids)
    ].copy()

    eval_df = feature_df[
        feature_df[
            "source1_entity_id"
        ].isin(eval_ids)
    ].copy()

    columns = select_model_features(
        feature_df
    )

    X_train = train_df[
        columns
    ].copy()

    y_train = train_df[
        "label"
    ].copy()

    X_tune = tune_df[
        columns
    ].copy()

    y_tune = tune_df[
        "label"
    ].copy()

    X_eval = eval_df[
        columns
    ].copy()

    y_eval = eval_df[
        "label"
    ].copy()

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(
            X_train[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "Non-numeric model features detected: "
            f"{non_numeric}"
        )

    print(
        f"\nModel features: "
        f"{len(columns)}"
    )

    print(
        f"Train candidate rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Tune candidate rows: "
        f"{len(tune_df):,}"
    )

    print(
        f"Eval candidate rows: "
        f"{len(eval_df):,}"
    )

    # ---------------------------------------------------------------
    # 10. Train LightGBM.
    # ---------------------------------------------------------------

    print(
        "\nTraining LightGBM..."
    )

    model = LightGBMMatcher(
        params={
            "objective": "binary",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        }
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_tune,
            y_tune,
        ),
    )

    # ---------------------------------------------------------------
    # 11. Threshold tuning.
    # ---------------------------------------------------------------

    print(
        "\nScoring threshold-tuning set..."
    )

    tune_probabilities = (
        model.predict_proba(
            X_tune
        )
    )

    tune_scores = (
        build_score_mapping(
            tune_df[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ],
            tune_probabilities,
        )
    )

    # Include entities with zero candidate rows.
    for entity_id in tune_ids:
        tune_scores.setdefault(
            entity_id,
            {},
        )

    tune_ground_truth = {
        entity_id:
            sampled_ground_truth.get(
                entity_id,
                set(),
            )
        for entity_id in tune_ids
    }

    threshold_result = (
        search_best_threshold(
            tune_ground_truth,
            tune_scores,
        )
    )

    threshold = (
        threshold_result.threshold
    )

    print(
        f"Selected threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Tuning F0.5: "
        f"{threshold_result.score:.6f}"
    )

    # ---------------------------------------------------------------
    # 12. Held-out evaluation.
    # ---------------------------------------------------------------

    print(
        "\nEvaluating unseen evaluation set..."
    )

    eval_probabilities = (
        model.predict_proba(
            X_eval
        )
    )

    eval_scores = (
        build_score_mapping(
            eval_df[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ],
            eval_probabilities,
        )
    )

    # Include entities with zero candidate rows.
    for entity_id in eval_ids:
        eval_scores.setdefault(
            entity_id,
            {},
        )

    predictions = {
        entity_id: {
            candidate_id
            for candidate_id, probability
            in scores.items()
            if probability >= threshold
        }
        for entity_id, scores
        in eval_scores.items()
    }

    eval_ground_truth = {
        entity_id:
            sampled_ground_truth.get(
                entity_id,
                set(),
            )
        for entity_id in eval_ids
    }

    evaluation_score = macro_f05(
        eval_ground_truth,
        predictions,
    )

    print(
        f"\nHELD-OUT F0.5: "
        f"{evaluation_score:.6f}"
    )

    # ---------------------------------------------------------------
    # 13. Feature importance.
    # ---------------------------------------------------------------

    importance = model.feature_importance(
        importance_type="gain"
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 14. Save model.
    # ---------------------------------------------------------------

    model_path = (
        ARTIFACT_DIR
        / "submission2_lightgbm.joblib"
    )

    metadata_path = (
        ARTIFACT_DIR
        / "submission2_metadata.json"
    )

    model.save(
        model_path
    )

    metadata = {
        "s1_sample":
            S1_SAMPLE,

        "blocking_frequency":
            MAX_BLOCKING_FREQUENCY,

        "candidate_pairs":
            int(
                len(candidate_df)
            ),

        "link_recall":
            float(
                link_recall
            ),

        "complete_entity_recall":
            float(
                complete_recall
            ),

        "threshold":
            float(
                threshold
            ),

        "tuning_f05":
            float(
                threshold_result.score
            ),

        "held_out_f05":
            float(
                evaluation_score
            ),

        "feature_count":
            len(columns),

        "features":
            columns,

        "candidate_artifact":
            str(
                candidate_artifact
            ),

        "feature_artifact":
            str(
                feature_artifact
            ),

        "random_seed":
            RANDOM_SEED,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        f"\nSaved model:"
        f"\n  {model_path}"
    )

    print(
        f"Saved metadata:"
        f"\n  {metadata_path}"
    )

    print(
        "\nSaved reusable artifacts:"
        f"\n  {candidate_artifact}"
        f"\n  {feature_artifact}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SUBMISSION #2 TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()