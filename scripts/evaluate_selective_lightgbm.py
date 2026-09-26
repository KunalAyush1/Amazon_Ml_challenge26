from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------
# Paths
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


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------


def load_ground_truth() -> dict[str, set[str]]:
    """Load and normalize ground-truth mapping."""

    raw = read_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    return {
        str(entity_id): set(matches or [])
        for entity_id, matches in raw.items()
    }


# ---------------------------------------------------------------------
# Deterministic S1 split
# ---------------------------------------------------------------------


def get_sample_s1_ids() -> list[str]:
    """Recover the exact 2,000-S1 sample."""

    con = duckdb.connect()

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

    con.close()

    return [
        str(row[0])
        for row in rows
    ]


def split_s1_entities(
    s1_ids: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """70% train, 15% threshold tuning, 15% held-out evaluation."""

    groups = pd.DataFrame(
        {
            "source1_entity_id": s1_ids
        }
    )

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )

    train_idx, temp_idx = next(
        first.split(
            groups,
            groups=groups["source1_entity_id"],
        )
    )

    train_ids = set(
        groups.iloc[train_idx]["source1_entity_id"]
    )

    temp = groups.iloc[temp_idx].copy()

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    tune_idx, eval_idx = next(
        second.split(
            temp,
            groups=temp["source1_entity_id"],
        )
    )

    tune_ids = set(
        temp.iloc[tune_idx]["source1_entity_id"]
    )

    eval_ids = set(
        temp.iloc[eval_idx]["source1_entity_id"]
    )

    assert train_ids.isdisjoint(tune_ids)
    assert train_ids.isdisjoint(eval_ids)
    assert tune_ids.isdisjoint(eval_ids)

    return (
        {str(x) for x in train_ids},
        {str(x) for x in tune_ids},
        {str(x) for x in eval_ids},
    )


# ---------------------------------------------------------------------
# Build complete candidate records
# ---------------------------------------------------------------------


def load_candidate_pairs() -> pd.DataFrame:
    """Load selective-prefix candidate IDs."""

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found:\n{CANDIDATE_PATH}"
        )

    df = pd.read_parquet(
        CANDIDATE_PATH
    )

    required = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate artifact missing columns: {sorted(missing)}"
        )

    df = df[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].drop_duplicates()

    df["source1_entity_id"] = (
        df["source1_entity_id"].astype(str)
    )

    df["candidate_entity_id"] = (
        df["candidate_entity_id"].astype(str)
    )

    return df


def build_candidate_records(
    candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join candidate IDs to normalized S1/S2/S3 records.

    Only the 2,000-S1 experiment population is loaded into the
    resulting dataframe, keeping the experiment manageable.
    """

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

    # Register the candidate dataframe.
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

    # Deterministic S1 sample.
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

    # Full target views.
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
            address_numbers

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
            address_numbers

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
                AS candidate_address_numbers

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
    """Generate pairwise model features."""

    print(
        "\nBuilding pair features..."
    )

    rows: list[dict[str, object]] = []

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

        output = {
            "source1_entity_id":
                str(
                    row["source1_entity_id"]
                ),

            "candidate_entity_id":
                str(
                    row["candidate_entity_id"]
                ),

            "candidate_is_s2":
                int(
                    str(
                        row["candidate_entity_id"]
                    ).startswith("S2-")
                ),

            "candidate_is_s3":
                int(
                    str(
                        row["candidate_entity_id"]
                    ).startswith("S3-")
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
# Feature selection
# ---------------------------------------------------------------------


def select_model_features(
    df: pd.DataFrame,
) -> list[str]:
    """Use the same feature family as the successful baseline."""

    excluded = {
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "label",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
        and not column.startswith("blocker_")
    ]

    return features


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
        for source1_id, candidate_id
        in zip(
            feature_df["source1_entity_id"],
            feature_df["candidate_entity_id"],
        )
    ]

    return feature_df


# ---------------------------------------------------------------------
# Score mapping
# ---------------------------------------------------------------------


def build_score_mapping(
    pair_df: pd.DataFrame,
    probabilities,
) -> dict[str, dict[str, float]]:
    """Build entity -> candidate -> probability mapping."""

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for row, probability in zip(
        pair_df.itertuples(index=False),
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
        "SELECTIVE PREFIX + LIGHTGBM EVALUATION"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Load ground truth and candidates.
    # ---------------------------------------------------------------

    ground_truth = load_ground_truth()

    candidate_pairs = (
        load_candidate_pairs()
    )

    sample_s1_ids = get_sample_s1_ids()

    print(
        f"\nCandidate rows: "
        f"{len(candidate_pairs):,}"
    )

    print(
        f"Sample S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    # ---------------------------------------------------------------
    # 2. Join candidate records.
    # ---------------------------------------------------------------

    candidate_records = (
        build_candidate_records(
            candidate_pairs
        )
    )

    # ---------------------------------------------------------------
    # 3. Features.
    # ---------------------------------------------------------------

    feature_df = build_features(
        candidate_records
    )

    # ---------------------------------------------------------------
    # 4. Labels.
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
    # 5. Group split.
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
    ]

    y_train = train_df[
        "label"
    ]

    X_tune = tune_df[
        columns
    ]

    y_tune = tune_df[
        "label"
    ]

    X_eval = eval_df[
        columns
    ]

    y_eval = eval_df[
        "label"
    ]

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
    # 6. Train LightGBM.
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
    # 7. Tune threshold.
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

    for entity_id in tune_ids:
        tune_scores.setdefault(
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

    threshold_result = (
        search_best_threshold(
            tune_truth,
            tune_scores,
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

    # ---------------------------------------------------------------
    # 8. Held-out evaluation.
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

    for entity_id in eval_ids:
        eval_scores.setdefault(
            entity_id,
            {},
        )

    eval_predictions = {
        entity_id: {
            candidate_id
            for candidate_id, probability
            in scores.items()
            if probability >= threshold
        }
        for entity_id, scores
        in eval_scores.items()
    }

    eval_truth = {
        entity_id:
            ground_truth.get(
                entity_id,
                set(),
            )
        for entity_id in eval_ids
    }

    held_out_f05 = macro_f05(
        eval_truth,
        eval_predictions,
    )

    print(
        f"\nHELD-OUT F0.5: "
        f"{held_out_f05:.6f}"
    )

    # ---------------------------------------------------------------
    # 9. Feature importance.
    # ---------------------------------------------------------------

    print(
        "\nTop 20 feature importances:"
    )

    print(
        model.feature_importance(
            importance_type="gain"
        ).head(20).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 10. Save model.
    # ---------------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    METADATA_PATH.write_text(
        json.dumps(
            {
                "candidate_artifact":
                    str(
                        CANDIDATE_PATH
                    ),

                "candidate_rows":
                    int(
                        len(candidate_pairs)
                    ),

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

                "feature_count":
                    len(columns),

                "features":
                    columns,

                "random_seed":
                    RANDOM_SEED,
            },
            indent=2,
        )
    )

    print(
        f"\nSaved model:"
        f"\n  {MODEL_PATH}"
    )

    print(
        f"Saved metadata:"
        f"\n  {METADATA_PATH}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SELECTIVE LIGHTGBM EVALUATION COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()