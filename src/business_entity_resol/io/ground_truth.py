from pathlib import Path

import pandas as pd


GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


class GroundTruthValidationError(ValueError):
    """Raised when ground-truth data is malformed."""


def parse_matched_ids(value: object) -> set[str]:
    """
    Convert a comma-separated match list into a set.

    Empty values represent no matches.
    """
    if value is None:
        return set()

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    ids = {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }

    for entity_id in ids:
        if not (
            entity_id.startswith("S2-")
            or entity_id.startswith("S3-")
        ):
            raise GroundTruthValidationError(
                f"Invalid matched entity ID: {entity_id}"
            )

    return ids


def read_ground_truth(
    path: str | Path,
) -> dict[str, set[str]]:
    """
    Read the challenge ground-truth TSV.

    Returns
    -------
    dict[str, set[str]]
        Mapping from Source-1 IDs to matching Source-2/Source-3 IDs.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    missing_columns = [
        column
        for column in GROUND_TRUTH_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise GroundTruthValidationError(
            f"Missing required ground-truth columns: {missing_columns}"
        )

    result: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        source1_id = str(row["source1_entity_id"]).strip()

        if not source1_id:
            raise GroundTruthValidationError(
                "Ground truth contains an empty source1_entity_id."
            )

        if not source1_id.startswith("S1-"):
            raise GroundTruthValidationError(
                f"Invalid Source-1 entity ID: {source1_id}"
            )

        if source1_id in result:
            raise GroundTruthValidationError(
                f"Duplicate source1_entity_id: {source1_id}"
            )

        result[source1_id] = parse_matched_ids(
            row["matched_entity_ids"]
        )

    return result