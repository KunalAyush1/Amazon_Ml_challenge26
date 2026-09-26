from __future__ import annotations

import json
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


from business_entity_resol.evaluation.entity_f05 import macro_f05
from business_entity_resol.evaluation.threshold_search import (
    search_best_threshold,
)
from business_entity_resol.features.pair_features import pair_features
from business_entity_resol.io.ground_truth import read_ground_truth
from business_entity_resol.models.lightgbm.model import LightGBMMatcher


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

CANDIDATE_PATH = (
    ARTIFACT_DIR
    / "selective_prefix_p7_a10_f100.parquet"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "submission2_selective_lightgbm.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "submission2_selective_metadata.json"
)

S1_SAMPLE = 2_000
RANDOM_SEED = 42

K_VALUES = [10, 20, 30]


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------


def load_ground_truth() -> dict[str, set[str]]:
    """Load ground truth as entity -> set(candidate_ids)."""

    raw = read_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    return {
        str(entity_id): set(matches or [])
        for entity_id, matches in raw.items()
    }


# ---------------------------------------------------------------------
# Deterministic Source-1 sample
# ---------------------------------------------------------------------


def get_sample_s1_ids() -> list[str]:
    """Recover the exact deterministic 2,000-S1 sample."""

    s1_glob = str(
        WORK_DIR
        / "train_source1"
        / "part_*.parquet"
    )

    con = duckdb.connect()

    rows = con.execute(
        f"""
        SELECT entity_id
        FROM read_parquet('{s1_glob}')
        ORDER BY hash(entity_id)
        LIMIT {S1_SAMPLE};
        """
    ).fetchall()

    con.close()

    return [
        str(row[0])
        for row in rows
    ]


# ---------------------------------------------------------------------
# Group split
# ---------------------------------------------------------------------


def split_s1_entities(
    s1_ids: list[str],
) -> tuple[
    set[str],
    set[str],
    set[str],
]:
    """
    Split S1 entities into:

        70% train
        15% threshold tuning
        15% held-out evaluation
    """

    groups = pd.DataFrame(
        {
            "source1_entity_id": s1_ids,
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
            groups=groups["source1_entity_id"],
        )
    )

    train_ids = set(
        groups.iloc[
            train_idx
        ]["source1_entity_id"].astype(str)
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
            groups=temp["source1_entity_id"],
        )
    )

    tune_ids = set(
        temp.iloc[
            tune_idx
        ]["source1_entity_id"].astype(str)
    )

    eval_ids = set(
        temp.iloc[
            eval_idx
        ]["source1_entity_id"].astype(str)
    )

    # Leakage safety checks.
    assert train_ids.isdisjoint(tune_ids)
    assert train_ids.isdisjoint(eval_ids)
    assert tune_ids.isdisjoint(eval_ids)

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
# Candidate loading
# ---------------------------------------------------------------------


def load_candidate_pairs() -> pd.DataFrame:
    """Load the selective-prefix candidate artifact."""

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found:\n"
            f"{CANDIDATE_PATH}"
        )

    df = pd.read_parquet(
        CANDIDATE_PATH
    )

    required_columns = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Candidate artifact is missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df[
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    df["source1_entity_id"] = (
        df["source1_entity_id"].astype(str)
    )

    df["candidate_entity_id"] = (
        df["candidate_entity_id"].astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Join candidate IDs to normalized records
# ---------------------------------------------------------------------


def build_candidate_records(
    candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate IDs to normalized S1/S2/S3 records."""

    print(
        "\nJoining candidate IDs to normalized records..."
    )

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

    # Register candidate dataframe.
    con.register(
        "candidate_pairs_df",
        candidate_pairs,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW candidate_pairs AS
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidate_pairs_df;
        """
    )

    # Exact deterministic S1 sample.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1 AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_compact,
            address_norm,
            address_compact,
            address_numbers

        FROM read_parquet(
            '{s1_glob}'
        )

        ORDER BY hash(entity_id)

        LIMIT {S1_SAMPLE};
        """
    )

    # Full targets.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s2 AS

        SELECT
            entity_id AS candidate_entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_compact,
            address_norm,
            address_compact,
            address_numbers,
            'S2' AS candidate_source

        FROM read_parquet(
            '{s2_glob}'
        );
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s3 AS

        SELECT
            entity_id AS candidate_entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            name_compact,
            address_norm,
            address_compact,
            address_numbers,
            'S3' AS candidate_source

        FROM read_parquet(
            '{s3_glob}'
        );
        """
    )

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

        FROM candidate_pairs cp

        INNER JOIN s1
            ON cp.source1_entity_id =
               s1.entity_id

        INNER JOIN targets t
            ON cp.candidate_entity_id =
               t.candidate_entity_id;
    """

    result = con.execute(
        query
    ).fetch_df()

    con.close()

    print(
        f"Joined candidate rows: "
        f"{len(result):,}"
    )

    return result


# ---------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------


def build_features(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the exact feature schema expected by LightGBM."""

    print(
        "\nBuilding pair features..."
    )

    rows: list[
        dict[str, object]
    ] = []

    records = candidate_df.to_dict(
        orient="records"
    )

    total = len(records)

    for index, row in enumerate(
        records,
        start=1,
    ):
        source1 = {
            "entity_id":
                row["source1_entity_id"],

            "business_name":
                row["source1_business_name"],

            "business_address":
                row["source1_business_address"],

            "country":
                row["source1_country"],

            "name_norm":
                row["source1_name_norm"],

            "name_compact":
                row["source1_name_compact"],

            "address_norm":
                row["source1_address_norm"],

            "address_compact":
                row["source1_address_compact"],

            "address_numbers":
                row["source1_address_numbers"],
        }

        candidate = {
            "entity_id":
                row["candidate_entity_id"],

            "business_name":
                row["candidate_business_name"],

            "business_address":
                row["candidate_business_address"],

            "country":
                row["candidate_country"],

            "name_norm":
                row["candidate_name_norm"],

            "name_compact":
                row["candidate_name_compact"],

            "address_norm":
                row["candidate_address_norm"],

            "address_compact":
                row["candidate_address_compact"],

            "address_numbers":
                row["candidate_address_numbers"],

        }

        features = pair_features(
            source1,
            candidate,
        )

        candidate_id = str(
            row["candidate_entity_id"]
        )

        # IMPORTANT:
        # These two columns are part of the saved LightGBM schema.
        output = {
            "source1_entity_id":
                str(
                    row[
                        "source1_entity_id"
                    ]
                ),

            "candidate_entity_id":
                candidate_id,

            "candidate_is_s2":
                int(
                    candidate_id.startswith(
                        "S2-"
                    )
                ),

            "candidate_is_s3":
                int(
                    candidate_id.startswith(
                        "S3-"
                    )
                ),
        }

        output.update(
            features
        )

        rows.append(
            output
        )

        if index % 20_000 == 0:
            print(
                f"  features: "
                f"{index:,}/{total:,}"
            )

    result = pd.DataFrame(
        rows
    )

    print(
        f"Completed "
        f"{len(result):,} feature rows."
    )

    return result


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------


def add_labels(
    df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
) -> pd.DataFrame:
    """Add binary pair labels."""

    df = df.copy()

    df["label"] = [
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
            df[
                "source1_entity_id"
            ],
            df[
                "candidate_entity_id"
            ],
        )
    ]

    return df


# ---------------------------------------------------------------------
# Heuristic pre-ranking
# ---------------------------------------------------------------------


def heuristic_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Cheap candidate pre-ranking.

    This is deliberately independent of LightGBM.
    """

    score = (
        1000.0
        * df[
            "name_norm_exact"
        ]

        +

        1000.0
        * df[
            "address_norm_exact"
        ]

        +

        100.0
        * df[
            "country_exact"
        ]

        +

        50.0
        * df[
            "numeric_jaccard"
        ]

        +

        30.0
        * df[
            "name_token_jaccard"
        ]

        +

        30.0
        * df[
            "address_token_jaccard"
        ]
    )

    return score


# ---------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------


def build_scores(
    df: pd.DataFrame,
    model: LightGBMMatcher,
    feature_names: list[str],
) -> pd.DataFrame:
    """Score a feature dataframe with the saved LightGBM model."""

    # Explicit schema verification.
    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Inference feature schema is missing: "
            f"{missing_features}"
        )

    X = df[
        feature_names
    ].copy()

    # Verify all model features are numeric.
    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "Non-numeric model features found: "
            f"{non_numeric}"
        )

    probabilities = model.predict_proba(
        X
    )

    output = df[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    output["probability"] = (
        probabilities
    )

    return output


# ---------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------


def make_mapping(
    df: pd.DataFrame,
) -> dict[
    str,
    dict[str, float],
]:
    """Create Source-1 -> candidate -> probability mapping."""

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for row in df.itertuples(
        index=False
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
            row.probability
        )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 72)
    print(
        "SELECTIVE PREFIX PRUNING + LIGHTGBM"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # Load artifacts
    # ---------------------------------------------------------------

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found:\n"
            f"{CANDIDATE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found:\n"
            f"{MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata artifact not found:\n"
            f"{METADATA_PATH}"
        )

    print(
        "\nLoading ground truth..."
    )

    ground_truth = (
        load_ground_truth()
    )

    print(
        "\nLoading candidate pairs..."
    )

    candidate_pairs = (
        load_candidate_pairs()
    )

    print(
        f"Candidate rows: "
        f"{len(candidate_pairs):,}"
    )

    # ---------------------------------------------------------------
    # Deterministic S1 split
    # ---------------------------------------------------------------

    sample_s1_ids = (
        get_sample_s1_ids()
    )

    (
        train_ids,
        tune_ids,
        eval_ids,
    ) = split_s1_entities(
        sample_s1_ids
    )

    print(
        f"Sample S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    print(
        f"Train entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Tune entities: "
        f"{len(tune_ids):,}"
    )

    print(
        f"Eval entities: "
        f"{len(eval_ids):,}"
    )

    # ---------------------------------------------------------------
    # Join candidate records
    # ---------------------------------------------------------------

    records = build_candidate_records(
        candidate_pairs
    )

    # ---------------------------------------------------------------
    # Build features
    # ---------------------------------------------------------------

    feature_df = build_features(
        records
    )

    # ---------------------------------------------------------------
    # Labels
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
    # Heuristic ranking score
    # ---------------------------------------------------------------

    print(
        "\nBuilding heuristic pre-ranking..."
    )

    feature_df[
        "heuristic_score"
    ] = heuristic_score(
        feature_df
    )

    # ---------------------------------------------------------------
    # Load saved model
    # ---------------------------------------------------------------

    print(
        "\nLoading selective LightGBM model..."
    )

    model = LightGBMMatcher.load(
        MODEL_PATH
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = json.load(
            handle
        )

    feature_names = list(
        metadata[
            "features"
        ]
    )

    print(
        f"Model features: "
        f"{len(feature_names)}"
    )

    # ---------------------------------------------------------------
    # Verify model schema BEFORE experiment
    # ---------------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Feature schema mismatch.\n"
            "Model expects:\n"
            f"{missing_features}\n"
            "Available columns:\n"
            f"{list(feature_df.columns)}"
        )

    # ---------------------------------------------------------------
    # K experiments
    # ---------------------------------------------------------------

    results: list[
        dict[str, object]
    ] = []

    for k in K_VALUES:

        print(
            "\n" + "-" * 64
        )

        print(
            f"K = {k}"
        )

        # -----------------------------------------------------------
        # Pre-rank candidates
        # -----------------------------------------------------------

        pruned = (
            feature_df
            .sort_values(
                [
                    "source1_entity_id",
                    "heuristic_score",
                    "candidate_entity_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
            .groupby(
                "source1_entity_id",
                sort=False,
            )
            .head(k)
            .reset_index(drop=True)
        )

        print(
            f"Candidate rows: "
            f"{len(pruned):,}"
        )

        # -----------------------------------------------------------
        # Tune subset
        # -----------------------------------------------------------

        tune_df = pruned[
            pruned[
                "source1_entity_id"
            ].isin(
                tune_ids
            )
        ].copy()

        print(
            f"Tune rows: "
            f"{len(tune_df):,}"
        )

        tune_scored = build_scores(
            tune_df,
            model,
            feature_names,
        )

        tune_mapping = make_mapping(
            tune_scored
        )

        # Include S1 entities with zero candidates.
        for entity_id in tune_ids:
            tune_mapping.setdefault(
                entity_id,
                {},
            )

        tune_truth = {
            entity_id:
                ground_truth.get(
                    entity_id,
                    set(),
                )
            for entity_id in tune_ids
        }

        # -----------------------------------------------------------
        # Threshold optimization
        # -----------------------------------------------------------

        threshold_result = (
            search_best_threshold(
                tune_truth,
                tune_mapping,
            )
        )

        threshold = float(
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

        # -----------------------------------------------------------
        # Held-out evaluation
        # -----------------------------------------------------------

        eval_df = pruned[
            pruned[
                "source1_entity_id"
            ].isin(
                eval_ids
            )
        ].copy()

        print(
            f"Eval rows: "
            f"{len(eval_df):,}"
        )

        eval_scored = build_scores(
            eval_df,
            model,
            feature_names,
        )

        eval_mapping = make_mapping(
            eval_scored
        )

        for entity_id in eval_ids:
            eval_mapping.setdefault(
                entity_id,
                {},
            )

        eval_truth = {
            entity_id:
                ground_truth.get(
                    entity_id,
                    set(),
                )
            for entity_id in eval_ids
        }

        predictions = {
            entity_id: {
                candidate_id
                for candidate_id, probability
                in eval_mapping[
                    entity_id
                ].items()
                if probability >= threshold
            }
            for entity_id in eval_ids
        }

        held_out_f05 = macro_f05(
            eval_truth,
            predictions,
        )

        print(
            f"HELD-OUT F0.5: "
            f"{held_out_f05:.6f}"
        )

        # -----------------------------------------------------------
        # Candidate count
        # -----------------------------------------------------------

        counts = (
            pruned
            .groupby(
                "source1_entity_id"
            )
            .size()
            .reindex(
                sample_s1_ids,
                fill_value=0,
            )
        )

        mean_candidates = float(
            counts.mean()
        )

        p95_candidates = float(
            counts.quantile(0.95)
        )

        zero_candidates = int(
            (counts == 0).sum()
        )

        print(
            f"Mean candidates/S1: "
            f"{mean_candidates:.3f}"
        )

        print(
            f"P95 candidates/S1: "
            f"{p95_candidates:.3f}"
        )

        print(
            f"Zero-candidate S1: "
            f"{zero_candidates:,}"
        )

        results.append(
            {
                "K": k,
                "candidate_rows":
                    int(len(pruned)),
                "tune_rows":
                    int(len(tune_df)),
                "eval_rows":
                    int(len(eval_df)),
                "threshold":
                    threshold,
                "tuning_f05":
                    float(
                        threshold_result.score
                    ),
                "held_out_f05":
                    float(
                        held_out_f05
                    ),
                "mean_candidates":
                    mean_candidates,
                "p95_candidates":
                    p95_candidates,
                "zero_candidate_s1":
                    zero_candidates,
            }
        )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    output_path = (
        ARTIFACT_DIR
        / "submission2_selective_pruning_f05.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL PRUNING RESULTS"
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
        f"\nSaved:"
        f"\n  {output_path}"
    )


if __name__ == "__main__":
    main()