from pathlib import Path
import random
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resol.preprocessing.normalization import normalize_dataframe
from business_entity_resol.blocking.candidate_generator import CandidateGenerator
from business_entity_resol.evaluation.blocking_evaluation import (
    evaluate_candidate_union,
)
from business_entity_resol.io.ground_truth import read_ground_truth


TRAIN_DIR = PROJECT_ROOT / "data" / "train"

S1_SAMPLE = 500
EXTRA_S2 = 5_000
EXTRA_S3 = 5_000

RANDOM_SEED = 42


def read_first_s1_records() -> pd.DataFrame:
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
    """
    Read complete source file and select:
      - all required ground-truth IDs
      - additional random negative records
    """
    path = TRAIN_DIR / filename

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
    )

    required = df[df["entity_id"].isin(required_ids)]

    if len(required) != len(required_ids):
        found = set(required["entity_id"])
        missing = required_ids - found

        raise ValueError(
            f"{filename}: {len(missing)} required IDs were not found. "
            f"Examples: {sorted(missing)[:10]}"
        )

    remaining = df[~df["entity_id"].isin(required_ids)]

    sample_size = min(extra_count, len(remaining))

    extra = remaining.sample(
        n=sample_size,
        random_state=seed,
    )

    return pd.concat(
        [required, extra],
        ignore_index=True,
    )


def main() -> None:
    random.seed(RANDOM_SEED)

    print("Reading ground truth...")

    ground_truth = read_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print("Selecting S1 sample...")

    source1 = read_first_s1_records()

    sampled_ids = set(source1["entity_id"])

    sampled_ground_truth = {
        entity_id: matches
        for entity_id, matches in ground_truth.items()
        if entity_id in sampled_ids
    }

    print(f"S1 sample: {len(source1):,}")
    print(
        "S1 records with ground truth: "
        f"{len(sampled_ground_truth):,}"
    )

    required_s2 = {
        entity_id
        for matches in sampled_ground_truth.values()
        for entity_id in matches
        if entity_id.startswith("S2-")
    }

    required_s3 = {
        entity_id
        for matches in sampled_ground_truth.values()
        for entity_id in matches
        if entity_id.startswith("S3-")
    }

    print(f"Required S2 matches: {len(required_s2):,}")
    print(f"Required S3 matches: {len(required_s3):,}")

    print("\nLoading S2 records...")

    source2 = read_records_by_ids(
        "train_source2.tsv",
        required_s2,
        EXTRA_S2,
        RANDOM_SEED,
    )

    print(f"S2 experiment records: {len(source2):,}")

    print("\nLoading S3 records...")

    source3 = read_records_by_ids(
        "train_source3.tsv",
        required_s3,
        EXTRA_S3,
        RANDOM_SEED + 1,
    )

    print(f"S3 experiment records: {len(source3):,}")

    print("\nNormalizing records...")

    source1 = normalize_dataframe(source1)
    source2 = normalize_dataframe(source2)
    source3 = normalize_dataframe(source3)

    print("Normalization complete.")

    s1_records = source1.to_dict("records")
    s2_records = source2.to_dict("records")
    s3_records = source3.to_dict("records")

    print("\nFitting candidate blockers...")

    generator = CandidateGenerator()
    generator.fit(
        s2_records,
        s3_records,
    )

    print("Generating candidates...")

    candidate_union = generator.generate(
        s1_records,
    )

    print(
        f"Total unique candidates: "
        f"{len(candidate_union):,}"
    )

    print("\nEvaluating blocking...")

    metrics = evaluate_candidate_union(
        sampled_ground_truth,
        candidate_union,
    )

    print("\n========== BLOCKING RESULTS ==========")

    print(
        f"Link recall:             "
        f"{metrics['link_recall']:.4f}"
    )

    print(
        f"Complete entity recall:  "
        f"{metrics['complete_entity_recall']:.4f}"
    )

    print("\nCandidate count statistics:")

    for key, value in metrics["candidate_count_stats"].items():
        print(f"  {key}: {value}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()