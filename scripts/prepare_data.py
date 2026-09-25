from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.io.ground_truth import read_ground_truth
from business_entity_resol.io.reader import read_test_sources, read_train_sources
from business_entity_resol.io.writer import write_parquet
from business_entity_resol.preprocessing.normalization import normalize_dataframe


def save_processed_sources(
    sources: dict[str, object],
    output_dir: Path,
    split: str,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for source_name, df in sources.items():
        print(f"Normalizing {split} {source_name}: {len(df):,} rows")

        normalized_df = normalize_dataframe(df)

        output_path = split_dir / f"{split}_{source_name}.parquet"
        write_parquet(normalized_df, output_path)

        print(f"Saved: {output_path}")


def save_ground_truth(
    ground_truth_path: Path,
    output_dir: Path,
) -> None:
    print(f"Reading ground truth: {ground_truth_path}")

    ground_truth = read_ground_truth(ground_truth_path)

    rows = [
        {
            "source1_entity_id": source1_id,
            "matched_entity_ids": sorted(matched_ids),
        }
        for source1_id, matched_ids in ground_truth.items()
    ]

    import pandas as pd

    ground_truth_df = pd.DataFrame(rows)

    output_path = output_dir / "train" / "train_ground_truth.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_parquet(ground_truth_df, output_path)

    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and normalize Amazon ML Challenge datasets."
    )

    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing train_source*.tsv and train_ground_truth.tsv",
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test_source*.tsv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed Parquet files will be written",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training sources...")
    train_sources = read_train_sources(args.train_dir)

    print("Loading test sources...")
    test_sources = read_test_sources(args.test_dir)

    print("Preparing training sources...")
    save_processed_sources(
        train_sources,
        args.output_dir,
        "train",
    )

    print("Preparing test sources...")
    save_processed_sources(
        test_sources,
        args.output_dir,
        "test",
    )

    save_ground_truth(
        args.train_dir / "train_ground_truth.tsv",
        args.output_dir,
    )

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()