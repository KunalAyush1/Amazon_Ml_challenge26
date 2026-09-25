from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resol.preprocessing.normalization import normalize_dataframe
from business_entity_resol.blocking.candidate_generator import CandidateGenerator
from business_entity_resol.datasets.pair_builder import build_pair_dataset
from business_entity_resol.io.ground_truth import read_ground_truth


TRAIN_DIR = PROJECT_ROOT / "data" / "train"

S1_SAMPLE = 500
EXTRA_S2 = 5_000
EXTRA_S3 = 5_000


def read_s1_sample() -> pd.DataFrame:
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
    df = pd.read_csv(
        TRAIN_DIR / filename,
        sep="\t",
        dtype={"entity_id": "string"},
        keep_default_na=False,
    )

    required = df[df["entity_id"].isin(required_ids)]

    if len(required) != len(required_ids):
        found = set(required["entity_id"])
        missing = required_ids - found
        raise ValueError(
            f"{filename}: missing required IDs: "
            f"{sorted(missing)[:10]}"
        )

    remaining = df[~df["entity_id"].isin(required_ids)]

    extra = remaining.sample(
        n=min(extra_count, len(remaining)),
        random_state=seed,
    )

    return pd.concat(
        [required, extra],
        ignore_index=True,
    )


def main() -> None:
    print("Reading ground truth...")

    ground_truth = read_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print("Reading S1 sample...")

    source1 = read_s1_sample()

    sampled_ids = set(source1["entity_id"])

    sampled_ground_truth = {
        entity_id: matches
        for entity_id, matches in ground_truth.items()
        if entity_id in sampled_ids
    }

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

    print(f"S1: {len(source1):,}")
    print(f"Required S2 matches: {len(required_s2):,}")
    print(f"Required S3 matches: {len(required_s3):,}")

    print("\nLoading S2/S3 experiment records...")

    source2 = read_records_by_ids(
        "train_source2.tsv",
        required_s2,
        EXTRA_S2,
        42,
    )

    source3 = read_records_by_ids(
        "train_source3.tsv",
        required_s3,
        EXTRA_S3,
        43,
    )

    print(f"S2: {len(source2):,}")
    print(f"S3: {len(source3):,}")

    print("\nNormalizing...")

    source1 = normalize_dataframe(source1)
    source2 = normalize_dataframe(source2)
    source3 = normalize_dataframe(source3)

    s1_records = source1.to_dict("records")
    s2_records = source2.to_dict("records")
    s3_records = source3.to_dict("records")

    print("\nGenerating candidates...")

    generator = CandidateGenerator()
    generator.fit(s2_records, s3_records)

    candidate_union = generator.generate(s1_records)

    candidate_records = candidate_union.to_dicts()

    print(
        f"Candidate pairs: {len(candidate_records):,}"
    )

    print("\nBuilding candidate lookup...")

    candidate_lookup = {}

    for record in s2_records + s3_records:
        candidate_lookup[
            str(record["entity_id"])
        ] = record

    print(
        f"Candidate records available: "
        f"{len(candidate_lookup):,}"
    )

    print("\nBuilding training pairs...")

    rows = build_pair_dataset(
        s1_records,
        candidate_records,
        sampled_ground_truth,
        candidate_lookup,
        negative_ratio=1,
        random_seed=42,
    )

    print("\n========== PAIR DATASET RESULTS ==========")

    print(f"Training rows: {len(rows):,}")

    positive_count = sum(
        row["label"] == 1
        for row in rows
    )

    negative_count = sum(
        row["label"] == 0
        for row in rows
    )

    print(f"Positive rows: {positive_count:,}")
    print(f"Negative rows: {negative_count:,}")

    if rows:
        feature_names = {
            key
            for key in rows[0]
            if key not in {
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "label",
            }
        }

        print(
            f"Feature columns: {len(feature_names):,}"
        )

        print("\nExample positive row:")
        positive = next(
            (
                row
                for row in rows
                if row["label"] == 1
            ),
            None,
        )

        if positive:
            print(
                {
                    key: positive[key]
                    for key in list(positive)[:15]
                }
            )

        print("\nExample negative row:")
        negative = next(
            (
                row
                for row in rows
                if row["label"] == 0
            ),
            None,
        )

        if negative:
            print(
                {
                    key: negative[key]
                    for key in list(negative)[:15]
                }
            )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()