from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.blocking.candidate_generator import (
    CandidateGenerator,
)
from business_entity_resol.datasets.candidate_features import (
    build_candidate_feature_dataframe,
    feature_columns,
)
from business_entity_resol.evaluation.blocking_evaluation import (
    evaluate_candidate_union,
)
from business_entity_resol.evaluation.cross_validation import (
    group_train_val_split,
)
from business_entity_resol.evaluation.entity_f05 import (
    macro_f05,
)
from business_entity_resol.evaluation.threshold_search import (
    search_best_threshold,
)
from business_entity_resol.io.ground_truth import (
    read_ground_truth,
)
from business_entity_resol.models.lightgbm.train import (
    train_lightgbm_matcher,
)
from business_entity_resol.preprocessing.normalization import (
    normalize_dataframe,
)


TRAIN_DIR = PROJECT_ROOT / "data" / "train"

S1_SAMPLE = 500
EXTRA_S2 = 5_000
EXTRA_S3 = 5_000

RANDOM_SEED = 42


def read_s1_sample() -> pd.DataFrame:
    """Read a small Source-1 sample."""
    return pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
        nrows=S1_SAMPLE,
    )


def read_records_by_ids(
    filename: str,
    required_ids: set[str],
    extra_count: int,
    seed: int,
) -> pd.DataFrame:
    """Load required target records plus random distractors."""

    path = TRAIN_DIR / filename

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
    )

    required = df[
        df["entity_id"].isin(required_ids)
    ]

    found_ids = set(
        required["entity_id"]
    )

    missing = required_ids - found_ids

    if missing:
        raise ValueError(
            f"{filename}: missing {len(missing)} required IDs. "
            f"Examples: {sorted(missing)[:10]}"
        )

    remaining = df[
        ~df["entity_id"].isin(required_ids)
    ]

    extra = remaining.sample(
        n=min(extra_count, len(remaining)),
        random_state=seed,
    )

    return pd.concat(
        [required, extra],
        ignore_index=True,
    )


def build_candidate_scores(
    scored_pairs: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Convert scored pairs into threshold-search format."""

    result: dict[str, dict[str, float]] = {}

    for row in scored_pairs.itertuples(index=False):
        result.setdefault(
            str(row.source1_entity_id),
            {},
        )[str(row.candidate_entity_id)] = float(
            row.match_probability
        )

    return result


def select_feature_columns(
    df: pd.DataFrame,
    *,
    include_address: bool = True,
    include_lexical: bool = True,
    include_blocker: bool = True,
) -> list[str]:
    """Select model features by feature family.

    Parameters
    ----------
    df:
        Candidate feature dataframe.

    include_address:
        Include address-related features.

    include_lexical:
        Include lexical features.

    include_blocker:
        Include blocker-provenance features.
    """

    columns = feature_columns(df)

    selected: list[str] = []

    for column in columns:

        # Address family.
        if (
            not include_address
            and column.startswith("address_")
        ):
            continue

        # Lexical family.
        if not include_lexical:
            if (
                column.startswith("name_lexical_")
                or column.startswith("address_lexical_")
            ):
                continue

        # Blocker-provenance family.
        if (
            not include_blocker
            and column.startswith("blocker_")
        ):
            continue

        selected.append(column)

    return selected


def main() -> None:
    print("=" * 70)
    print("LIGHTGBM REAL-DATA BASELINE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Ground truth
    # ---------------------------------------------------------------
    print("\n[1/9] Loading ground truth...")

    ground_truth = read_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    # ---------------------------------------------------------------
    # 2. Source-1 sample
    # ---------------------------------------------------------------
    print("\n[2/9] Loading Source-1 sample...")

    source1 = read_s1_sample()

    source1_ids = set(
        source1["entity_id"]
    )

    sampled_ground_truth = {
        entity_id: matches
        for entity_id, matches in ground_truth.items()
        if entity_id in source1_ids
    }

    print(
        f"Source-1 sample: "
        f"{len(source1):,}"
    )

    print(
        f"S1 entities with ground truth: "
        f"{len(sampled_ground_truth):,}"
    )

    # ---------------------------------------------------------------
    # 3. S2/S3 experiment population
    # ---------------------------------------------------------------
    print(
        "\n[3/9] Loading Source-2/Source-3 experiment sets..."
    )

    required_s2 = {
        candidate_id
        for matches in sampled_ground_truth.values()
        for candidate_id in matches
        if candidate_id.startswith("S2-")
    }

    required_s3 = {
        candidate_id
        for matches in sampled_ground_truth.values()
        for candidate_id in matches
        if candidate_id.startswith("S3-")
    }

    print(
        f"Required S2 matches: "
        f"{len(required_s2):,}"
    )

    print(
        f"Required S3 matches: "
        f"{len(required_s3):,}"
    )

    source2 = read_records_by_ids(
        "train_source2.tsv",
        required_s2,
        EXTRA_S2,
        RANDOM_SEED,
    )

    source3 = read_records_by_ids(
        "train_source3.tsv",
        required_s3,
        EXTRA_S3,
        RANDOM_SEED + 1,
    )

    print(
        f"S2 experiment records: "
        f"{len(source2):,}"
    )

    print(
        f"S3 experiment records: "
        f"{len(source3):,}"
    )

    # ---------------------------------------------------------------
    # 4. Normalize
    # ---------------------------------------------------------------
    print("\n[4/9] Normalizing...")

    source1 = normalize_dataframe(
        source1
    )

    source2 = normalize_dataframe(
        source2
    )

    source3 = normalize_dataframe(
        source3
    )

    s1_records = source1.to_dict(
        orient="records"
    )

    s2_records = source2.to_dict(
        orient="records"
    )

    s3_records = source3.to_dict(
        orient="records"
    )

    # ---------------------------------------------------------------
    # 5. Candidate generation
    # ---------------------------------------------------------------
    print("\n[5/9] Generating candidates...")

    # Dense is disabled for the first LightGBM baseline.
    # This avoids the macOS PyTorch/LightGBM interaction and keeps
    # the baseline focused on the classical blockers.
    generator = CandidateGenerator(
        use_dense=False,
    )

    generator.fit(
        s2_records,
        s3_records,
    )

    candidate_union = generator.generate(
        s1_records
    )

    print(
        f"Unique candidate pairs: "
        f"{len(candidate_union):,}"
    )

    # ---------------------------------------------------------------
    # 6. Blocking evaluation
    # ---------------------------------------------------------------
    print("\n[6/9] Evaluating blocking...")

    blocking_metrics = evaluate_candidate_union(
        sampled_ground_truth,
        candidate_union,
    )

    print(
        f"Link recall: "
        f"{blocking_metrics['link_recall']:.6f}"
    )

    print(
        f"Complete entity recall: "
        f"{blocking_metrics['complete_entity_recall']:.6f}"
    )

    print("\nCandidate-count statistics:")

    for key, value in blocking_metrics[
        "candidate_count_stats"
    ].items():
        print(
            f"  {key}: {value}"
        )

    # ---------------------------------------------------------------
    # 7. Pair features
    # ---------------------------------------------------------------
    print("\n[7/9] Building pair features...")

    candidate_records = (
        candidate_union.to_dicts()
    )

    candidate_lookup = {
        str(record["entity_id"]): record
        for record in (
            s2_records + s3_records
        )
    }

    pair_df = build_candidate_feature_dataframe(
        source1_records=s1_records,
        candidate_records=candidate_records,
        candidate_lookup=candidate_lookup,
    )

    print(
        f"Feature rows: "
        f"{len(pair_df):,}"
    )

    # ---------------------------------------------------------------
    # 8. Labels + grouped split
    # ---------------------------------------------------------------
    print(
        "\n[8/9] Creating labels and validation split..."
    )

    pair_df["label"] = [
        int(
            candidate_id
            in sampled_ground_truth.get(
                source1_id,
                set(),
            )
        )
        for source1_id, candidate_id
        in zip(
            pair_df["source1_entity_id"],
            pair_df["candidate_entity_id"],
        )
    ]

    print("\nLabel distribution:")

    print(
        pair_df["label"].value_counts(
            dropna=False
        )
    )

    train_df, valid_df = group_train_val_split(
        pair_df,
        group_col="source1_entity_id",
        val_size=0.2,
        random_state=RANDOM_SEED,
    )

    print(
        f"\nTrain rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(valid_df):,}"
    )

    # Current full baseline:
    # address + lexical + blocker features.
    columns = select_feature_columns(
        pair_df,
        include_address=True,
        include_lexical=True,
        include_blocker=True,
    )

    X_train = train_df[
        columns
    ].copy()

    y_train = train_df[
        "label"
    ].copy()

    X_valid = valid_df[
        columns
    ].copy()

    y_valid = valid_df[
        "label"
    ].copy()

    # Ensure every model feature is numeric.
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
        f"Number of model features: "
        f"{len(columns)}"
    )

    # ---------------------------------------------------------------
    # 9. LightGBM + threshold optimization
    # ---------------------------------------------------------------
    print("\n[9/9] Training LightGBM...")

    result = train_lightgbm_matcher(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        },
    )

    model = result.model

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------
    print("\nTop 25 feature importances:")

    importance = model.feature_importance(
        importance_type="gain"
    )

    print(
        importance.head(25).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Validation scoring
    # ---------------------------------------------------------------
    print("\nScoring validation candidates...")

    validation_scores = model.predict_proba(
        X_valid
    )

    scored_valid = valid_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    scored_valid[
        "match_probability"
    ] = validation_scores

    candidate_scores = build_candidate_scores(
        scored_valid
    )

    # Every validation S1 entity must be represented.
    # This includes entities for which no candidate passes
    # the eventual threshold.
    validation_entities = set(
        valid_df[
            "source1_entity_id"
        ]
    )

    for entity_id in validation_entities:
        candidate_scores.setdefault(
            entity_id,
            {},
        )

    validation_ground_truth = {
        entity_id: sampled_ground_truth[
            entity_id
        ]
        for entity_id in validation_entities
    }

    # ---------------------------------------------------------------
    # Threshold optimization
    # ---------------------------------------------------------------
    print(
        "\nSearching for best threshold..."
    )

    threshold_result = search_best_threshold(
        validation_ground_truth,
        candidate_scores,
    )

    print(
        f"Best threshold: "
        f"{threshold_result.threshold:.4f}"
    )

    print(
        f"Validation macro F0.5: "
        f"{threshold_result.score:.6f}"
    )

    # ---------------------------------------------------------------
    # Independent verification using the selected threshold
    # ---------------------------------------------------------------
    predictions = {
        entity_id: {
            candidate_id
            for candidate_id, score in scores.items()
            if score >= threshold_result.threshold
        }
        for entity_id, scores in candidate_scores.items()
    }

    verified_score = macro_f05(
        validation_ground_truth,
        predictions,
    )

    print(
        f"Verified macro F0.5: "
        f"{verified_score:.6f}"
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()