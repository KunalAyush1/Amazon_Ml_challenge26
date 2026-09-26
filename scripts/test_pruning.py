from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.evaluation.entity_f05 import (
    macro_f05,
)
from business_entity_resol.io.ground_truth import (
    read_ground_truth,
)
from business_entity_resol.models.lightgbm.model import (
    LightGBMMatcher,
)


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
WORK_DIR = PROJECT_ROOT / "data" / "submission2_work"

CANDIDATE_PATH = (
    ARTIFACT_DIR
    / "submission2_candidates_2000.parquet"
)

FEATURE_PATH = (
    ARTIFACT_DIR
    / "submission2_features_2000.parquet"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "submission2_lightgbm.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "submission2_metadata.json"
)

TRAIN_DIR = PROJECT_ROOT / "data" / "train"

K_VALUES = [20, 30, 50, 100]


def get_sample_s1_ids() -> list[str]:
    """Reconstruct the exact deterministic 2,000-S1 sample."""

    con = duckdb.connect()

    s1_glob = str(
        WORK_DIR
        / "train_source1"
        / "part_*.parquet"
    )

    ids = [
        str(row[0])
        for row in con.execute(
            f"""
            SELECT entity_id
            FROM read_parquet('{s1_glob}')
            ORDER BY hash(entity_id)
            LIMIT 2000;
            """
        ).fetchall()
    ]

    con.close()

    return ids


def add_heuristic_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a cheap pre-ranking score."""

    df = df.copy()

    df["heuristic_score"] = (
        100.0
        * df["name_norm_exact"]
        +
        100.0
        * df["address_norm_exact"]
        +
        20.0
        * df["country_exact"]
        +
        10.0
        * df["numeric_jaccard"]
        +
        5.0
        * df["name_token_jaccard"]
        +
        5.0
        * df["address_token_jaccard"]
    )

    return df


def prune(
    df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Keep top-k candidates per Source-1."""

    ranked = (
        df.sort_values(
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

    return ranked


def evaluate_candidate_recall(
    df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    sample_s1_ids: list[str],
) -> tuple[float, float]:
    """Evaluate link and complete-entity recall."""

    candidates = (
        df.groupby(
            "source1_entity_id"
        )["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    total_true_links = 0
    retrieved_true_links = 0

    eligible_entities = 0
    complete_entities = 0

    for entity_id in sample_s1_ids:

        true_set = ground_truth.get(
            entity_id,
            set(),
        )

        if not true_set:
            continue

        eligible_entities += 1

        candidate_set = candidates.get(
            entity_id,
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


def evaluate_model(
    df: pd.DataFrame,
    model: LightGBMMatcher,
    feature_names: list[str],
    threshold: float,
    ground_truth: dict[str, set[str]],
    sample_s1_ids: list[str],
) -> float:
    """Score the pruned candidates and calculate entity-level F0.5."""

    X = df[
        feature_names
    ].copy()

    probabilities = model.predict_proba(
        X
    )

    df = df.copy()

    df["probability"] = probabilities

    scores: dict[
        str,
        dict[str, float],
    ] = {}

    for row in df.itertuples(
        index=False
    ):
        scores.setdefault(
            str(row.source1_entity_id),
            {},
        )[str(row.candidate_entity_id)] = float(
            row.probability
        )

    predictions: dict[
        str,
        set[str],
    ] = {}

    eval_truth: dict[
        str,
        set[str],
    ] = {}

    for entity_id in sample_s1_ids:

        entity_scores = scores.get(
            entity_id,
            {},
        )

        predictions[entity_id] = {
            candidate_id
            for candidate_id, probability
            in entity_scores.items()
            if probability >= threshold
        }

        eval_truth[entity_id] = ground_truth.get(
            entity_id,
            set(),
        )

    return macro_f05(
        eval_truth,
        predictions,
    )


def main() -> None:

    print("=" * 72)
    print("SUBMISSION #2 — PRUNING EXPERIMENT")
    print("=" * 72)

    print("\nLoading artifacts...")

    candidate_df = pd.read_parquet(
        CANDIDATE_PATH
    )

    feature_df = pd.read_parquet(
        FEATURE_PATH
    )

    model = LightGBMMatcher.load(
        MODEL_PATH
    )

    metadata = pd.read_json(
        METADATA_PATH,
        typ="series",
    )

    threshold = float(
        metadata["threshold"]
    )

    feature_names = list(
        metadata["features"]
    )

    print(
        f"Candidate rows: "
        f"{len(candidate_df):,}"
    )

    print(
        f"Feature rows: "
        f"{len(feature_df):,}"
    )

    print(
        f"Threshold: "
        f"{threshold:.4f}"
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
        for entity_id, matches
        in raw_ground_truth.items()
    }

    sample_s1_ids = get_sample_s1_ids()

    print(
        f"Sample S1 entities: "
        f"{len(sample_s1_ids):,}"
    )

    print(
        "\nCalculating heuristic scores..."
    )

    feature_df = add_heuristic_score(
        feature_df
    )

    # Keep only the same identity columns from candidate_df
    # in case feature_df changes in the future.
    working_df = feature_df.copy()

    results = []

    for k in K_VALUES:

        print(
            f"\n{'-' * 60}"
        )

        print(
            f"Testing K = {k}"
        )

        pruned = prune(
            working_df,
            k,
        )

        link_recall, complete_recall = (
            evaluate_candidate_recall(
                pruned,
                ground_truth,
                sample_s1_ids,
            )
        )

        f05 = evaluate_model(
            pruned,
            model,
            feature_names,
            threshold,
            ground_truth,
            sample_s1_ids,
        )

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

        mean_candidates = (
            float(counts.mean())
        )

        p95_candidates = (
            float(counts.quantile(0.95))
        )

        results.append(
            {
                "K": k,
                "candidate_rows": len(pruned),
                "link_recall": link_recall,
                "complete_entity_recall":
                    complete_recall,
                "mean_candidates":
                    mean_candidates,
                "p95_candidates":
                    p95_candidates,
                "held_out_f05":
                    f05,
            }
        )

        print(
            f"Candidate rows: "
            f"{len(pruned):,}"
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
            f"{p95_candidates:.3f}"
        )

        print(
            f"HELD-OUT F0.5: "
            f"{f05:.6f}"
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "PRUNING RESULTS"
    )

    print(
        "=" * 72
    )

    result_df = pd.DataFrame(
        results
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    output_path = (
        ARTIFACT_DIR
        / "submission2_pruning_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved:"
        f"\n  {output_path}"
    )


if __name__ == "__main__":
    main()