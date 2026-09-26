from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from business_entity_resol.models.lightgbm.model import (
    LightGBMMatcher,
)


# =====================================================================
# PATHS
# =====================================================================

WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "submission2_inference_work"
)

DB_PATH = (
    WORK_DIR
    / "submission2_inference.duckdb"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "submission2_selective_lightgbm.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "submission2_selective_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "submission2_fast"
)

CANDIDATE_OUTPUT = (
    OUTPUT_DIR
    / "candidate_pairs.tsv"
)

MATCHING_OUTPUT = (
    OUTPUT_DIR
    / "matching_results.tsv"
)

TEMP_DIR = (
    WORK_DIR
    / "fast_inference_tmp"
)


# =====================================================================
# RUNTIME CONFIGURATION
# =====================================================================

#
# Measured offline results:
#
# K=10 -> threshold 0.75 -> held-out F0.5 = 0.773051
# K=20 -> threshold 0.71 -> held-out F0.5 = 0.797043
# K=30 -> threshold 0.71 -> held-out F0.5 = 0.802502
#

TOP_K = int(
    os.environ.get(
        "SUBMISSION2_K",
        "10",
    )
)

THRESHOLD_BY_K = {
    10: 0.75,
    20: 0.71,
    30: 0.71,
}

if TOP_K not in THRESHOLD_BY_K:
    raise ValueError(
        "SUBMISSION2_K must be one of "
        f"{sorted(THRESHOLD_BY_K)}"
    )

THRESHOLD = float(
    os.environ.get(
        "SUBMISSION2_THRESHOLD",
        str(THRESHOLD_BY_K[TOP_K]),
    )
)

# Number of candidate rows fetched for feature computation at once.
FEATURE_BATCH_ROWS = 50_000

# Source-1 entities per inference batch.
S1_BATCH_SIZE = 5_000

# DuckDB parallelism.
DUCKDB_THREADS = max(
    1,
    min(
        8,
        os.cpu_count() or 4,
    ),
)

# RapidFuzz releases the GIL and can use all CPUs.
RAPIDFUZZ_WORKERS = -1


# =====================================================================
# DUCKDB CONFIGURATION
# =====================================================================


def configure_duckdb(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Configure DuckDB for the existing large database."""

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con.execute(
        f"PRAGMA threads={DUCKDB_THREADS}"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    con.execute(
        "SET memory_limit='10GB'"
    )

    con.execute(
        "SET max_temp_directory_size='50GB'"
    )

    con.execute(
        f"SET temp_directory='{TEMP_DIR}'"
    )


# =====================================================================
# TEXT / RAPIDFUZZ HELPERS
# =====================================================================


def clean_text_series(
    series: pd.Series,
) -> np.ndarray:
    """Convert a pandas series to a string object ndarray."""

    return (
        series
        .fillna("")
        .astype(str)
        .to_numpy(
            dtype=object
        )
    )


def rapid_ratio(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized fuzz.ratio."""

    return (
        process.cpdist(
            left,
            right,
            scorer=fuzz.ratio,
            workers=RAPIDFUZZ_WORKERS,
            dtype=np.float32,
        )
        / 100.0
    )


def rapid_partial_ratio(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized fuzz.partial_ratio."""

    return (
        process.cpdist(
            left,
            right,
            scorer=fuzz.partial_ratio,
            workers=RAPIDFUZZ_WORKERS,
            dtype=np.float32,
        )
        / 100.0
    )


def rapid_token_sort_ratio(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized fuzz.token_sort_ratio."""

    return (
        process.cpdist(
            left,
            right,
            scorer=fuzz.token_sort_ratio,
            workers=RAPIDFUZZ_WORKERS,
            dtype=np.float32,
        )
        / 100.0
    )


def rapid_token_set_ratio(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized fuzz.token_set_ratio."""

    return (
        process.cpdist(
            left,
            right,
            scorer=fuzz.token_set_ratio,
            workers=RAPIDFUZZ_WORKERS,
            dtype=np.float32,
        )
        / 100.0
    )


def rapid_jaro_winkler(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Vectorized Jaro-Winkler."""

    return process.cpdist(
        left,
        right,
        scorer=JaroWinkler.normalized_similarity,
        workers=RAPIDFUZZ_WORKERS,
        dtype=np.float32,
    )


# =====================================================================
# TOKEN STATISTICS
# =====================================================================


def token_statistics(
    left: pd.Series,
    right: pd.Series,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate:

        token_count_1
        token_count_2
        token_overlap_count
        token_jaccard
    """

    sets_left = [
        set(str(value).split())
        for value in left.fillna("")
    ]

    sets_right = [
        set(str(value).split())
        for value in right.fillna("")
    ]

    count_left = np.fromiter(
        (
            len(value)
            for value in sets_left
        ),
        dtype=np.int16,
        count=len(sets_left),
    )

    count_right = np.fromiter(
        (
            len(value)
            for value in sets_right
        ),
        dtype=np.int16,
        count=len(sets_right),
    )

    overlap = np.fromiter(
        (
            len(a & b)
            for a, b in zip(
                sets_left,
                sets_right,
            )
        ),
        dtype=np.int16,
        count=len(sets_left),
    )

    union = np.fromiter(
        (
            len(a | b)
            for a, b in zip(
                sets_left,
                sets_right,
            )
        ),
        dtype=np.int16,
        count=len(sets_left),
    )

    jaccard = np.divide(
        overlap.astype(np.float32),
        union.astype(np.float32),
        out=np.zeros(
            len(overlap),
            dtype=np.float32,
        ),
        where=union != 0,
    )

    return (
        count_left,
        count_right,
        overlap,
        jaccard,
    )


# =====================================================================
# NUMERIC STATISTICS
# =====================================================================


def numeric_statistics(
    left: pd.Series,
    right: pd.Series,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate numeric evidence features."""

    left_sets: list[set[str]] = []
    right_sets: list[set[str]] = []

    for value in left:
        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            left_sets.append(
                {
                    str(item).strip()
                    for item in value
                    if item is not None
                    and str(item).strip()
                }
            )
        else:
            left_sets.append(set())

    for value in right:
        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            right_sets.append(
                {
                    str(item).strip()
                    for item in value
                    if item is not None
                    and str(item).strip()
                }
            )
        else:
            right_sets.append(set())

    count_1 = np.fromiter(
        (
            len(value)
            for value in left_sets
        ),
        dtype=np.int16,
        count=len(left_sets),
    )

    count_2 = np.fromiter(
        (
            len(value)
            for value in right_sets
        ),
        dtype=np.int16,
        count=len(right_sets),
    )

    overlap = np.fromiter(
        (
            len(a & b)
            for a, b in zip(
                left_sets,
                right_sets,
            )
        ),
        dtype=np.int16,
        count=len(left_sets),
    )

    union = np.fromiter(
        (
            len(a | b)
            for a, b in zip(
                left_sets,
                right_sets,
            )
        ),
        dtype=np.int16,
        count=len(left_sets),
    )

    jaccard = np.divide(
        overlap.astype(np.float32),
        union.astype(np.float32),
        out=np.zeros(
            len(overlap),
            dtype=np.float32,
        ),
        where=union != 0,
    )

    exact_match = (
        (count_1 > 0)
        &
        (count_1 == count_2)
        &
        (overlap == count_1)
    ).astype(np.int8)

    any_overlap = (
        overlap > 0
    ).astype(np.int8)

    return (
        count_1,
        count_2,
        overlap,
        jaccard,
        exact_match,
        any_overlap,
    )


# =====================================================================
# PREPARE DERIVED FEATURES
# =====================================================================


def prepare_batch(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cheap/derived features used by the vectorized feature builder.
    """

    # ---------------------------------------------------------------
    # Name comparisons
    # ---------------------------------------------------------------

    name_norm_1 = (
        df[
            "source1_name_norm"
        ]
        .fillna("")
        .astype(str)
    )

    name_norm_2 = (
        df[
            "candidate_name_norm"
        ]
        .fillna("")
        .astype(str)
    )

    raw_name_1 = (
        df[
            "source1_business_name"
        ]
        .fillna("")
        .astype(str)
    )

    raw_name_2 = (
        df[
            "candidate_business_name"
        ]
        .fillna("")
        .astype(str)
    )

    name_compare_1 = name_norm_1.where(
        name_norm_1 != "",
        raw_name_1,
    )

    name_compare_2 = name_norm_2.where(
        name_norm_2 != "",
        raw_name_2,
    )

    name_compact_1 = (
        df[
            "source1_name_compact"
        ]
        .fillna("")
        .astype(str)
    )

    name_compact_2 = (
        df[
            "candidate_name_compact"
        ]
        .fillna("")
        .astype(str)
    )

    name_compact_compare_1 = (
        name_compact_1.where(
            name_compact_1 != "",
            name_compare_1.str.replace(
                " ",
                "",
                regex=False,
            ),
        )
    )

    name_compact_compare_2 = (
        name_compact_2.where(
            name_compact_2 != "",
            name_compare_2.str.replace(
                " ",
                "",
                regex=False,
            ),
        )
    )

    # ---------------------------------------------------------------
    # Name lengths
    # ---------------------------------------------------------------

    df[
        "name_length_1"
    ] = (
        name_compare_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_length_2"
    ] = (
        name_compare_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_length_diff"
    ] = (
        (
            df[
                "name_length_1"
            ]
            -
            df[
                "name_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    df[
        "name_compact_length_1"
    ] = (
        name_compact_compare_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_compact_length_2"
    ] = (
        name_compact_compare_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_compact_length_diff"
    ] = (
        (
            df[
                "name_compact_length_1"
            ]
            -
            df[
                "name_compact_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    # ---------------------------------------------------------------
    # Address comparisons
    # ---------------------------------------------------------------

    address_norm_1 = (
        df[
            "source1_address_norm"
        ]
        .fillna("")
        .astype(str)
    )

    address_norm_2 = (
        df[
            "candidate_address_norm"
        ]
        .fillna("")
        .astype(str)
    )

    raw_address_1 = (
        df[
            "source1_business_address"
        ]
        .fillna("")
        .astype(str)
    )

    raw_address_2 = (
        df[
            "candidate_business_address"
        ]
        .fillna("")
        .astype(str)
    )

    address_compare_1 = (
        address_norm_1.where(
            address_norm_1 != "",
            raw_address_1,
        )
    )

    address_compare_2 = (
        address_norm_2.where(
            address_norm_2 != "",
            raw_address_2,
        )
    )

    address_compact_1 = (
        df[
            "source1_address_compact"
        ]
        .fillna("")
        .astype(str)
    )

    address_compact_2 = (
        df[
            "candidate_address_compact"
        ]
        .fillna("")
        .astype(str)
    )

    address_compact_compare_1 = (
        address_compact_1.where(
            address_compact_1 != "",
            address_compare_1.str.replace(
                " ",
                "",
                regex=False,
            ),
        )
    )

    address_compact_compare_2 = (
        address_compact_2.where(
            address_compact_2 != "",
            address_compare_2.str.replace(
                " ",
                "",
                regex=False,
            ),
        )
    )

    # ---------------------------------------------------------------
    # Address missingness
    # ---------------------------------------------------------------

    df[
        "address_missing_1"
    ] = (
        address_compare_1 == ""
    ).astype(np.int8)

    df[
        "address_missing_2"
    ] = (
        address_compare_2 == ""
    ).astype(np.int8)

    df[
        "address_both_missing"
    ] = (
        (
            df[
                "address_missing_1"
            ]
            == 1
        )
        &
        (
            df[
                "address_missing_2"
            ]
            == 1
        )
    ).astype(np.int8)

    # ---------------------------------------------------------------
    # Address lengths
    # ---------------------------------------------------------------

    df[
        "address_length_1"
    ] = (
        address_compare_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_length_2"
    ] = (
        address_compare_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_length_diff"
    ] = (
        (
            df[
                "address_length_1"
            ]
            -
            df[
                "address_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    df[
        "address_compact_length_1"
    ] = (
        address_compact_compare_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_compact_length_2"
    ] = (
        address_compact_compare_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_compact_length_diff"
    ] = (
        (
            df[
                "address_compact_length_1"
            ]
            -
            df[
                "address_compact_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    # ---------------------------------------------------------------
    # Name lexical token statistics
    # ---------------------------------------------------------------

    (
        name_token_count_1,
        name_token_count_2,
        name_token_overlap,
        name_token_jaccard,
    ) = token_statistics(
        name_norm_1,
        name_norm_2,
    )

    df[
        "name_lexical_token_count_1"
    ] = name_token_count_1

    df[
        "name_lexical_token_count_2"
    ] = name_token_count_2

    df[
        "name_lexical_token_overlap_count"
    ] = name_token_overlap

    df[
        "name_lexical_token_jaccard"
    ] = name_token_jaccard

    df[
        "name_lexical_length_1"
    ] = (
        name_norm_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_lexical_length_2"
    ] = (
        name_norm_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "name_lexical_length_diff"
    ] = (
        (
            df[
                "name_lexical_length_1"
            ]
            -
            df[
                "name_lexical_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    # ---------------------------------------------------------------
    # Address lexical token statistics
    # ---------------------------------------------------------------

    (
        address_token_count_1,
        address_token_count_2,
        address_token_overlap,
        address_token_jaccard,
    ) = token_statistics(
        address_norm_1,
        address_norm_2,
    )

    df[
        "address_lexical_token_count_1"
    ] = address_token_count_1

    df[
        "address_lexical_token_count_2"
    ] = address_token_count_2

    df[
        "address_lexical_token_overlap_count"
    ] = address_token_overlap

    df[
        "address_lexical_token_jaccard"
    ] = address_token_jaccard

    df[
        "address_lexical_length_1"
    ] = (
        address_norm_1
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_lexical_length_2"
    ] = (
        address_norm_2
        .str.len()
        .astype(np.int32)
    )

    df[
        "address_lexical_length_diff"
    ] = (
        (
            df[
                "address_lexical_length_1"
            ]
            -
            df[
                "address_lexical_length_2"
            ]
        )
        .abs()
        .astype(np.int32)
    )

    # ---------------------------------------------------------------
    # Numeric features
    # ---------------------------------------------------------------

    (
        numeric_count_1,
        numeric_count_2,
        numeric_overlap,
        numeric_jaccard,
        numeric_exact,
        numeric_any_overlap,
    ) = numeric_statistics(
        df[
            "source1_address_numbers"
        ],
        df[
            "candidate_address_numbers"
        ],
    )

    df[
        "numeric_count_1"
    ] = numeric_count_1

    df[
        "numeric_count_2"
    ] = numeric_count_2

    df[
        "numeric_overlap_count"
    ] = numeric_overlap

    df[
        "numeric_jaccard"
    ] = numeric_jaccard

    df[
        "numeric_exact_match"
    ] = numeric_exact

    df[
        "numeric_any_overlap"
    ] = numeric_any_overlap

    # ---------------------------------------------------------------
    # Country features
    # ---------------------------------------------------------------

    country1 = (
        df[
            "source1_country"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    country2 = (
        df[
            "candidate_country"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    country_equal = (
        (
            country1 != ""
        )
        &
        (
            country2 != ""
        )
        &
        (
            country1 == country2
        )
    )

    df[
        "country_exact"
    ] = country_equal.astype(
        np.int8
    )

    df[
        "country_norm_exact"
    ] = country_equal.astype(
        np.int8
    )

    df[
        "country_code_exact"
    ] = country_equal.astype(
        np.int8
    )

    df[
        "country_both_present"
    ] = (
        (
            country1 != ""
        )
        &
        (
            country2 != ""
        )
    ).astype(np.int8)

    df[
        "country_missing_1"
    ] = (
        country1 == ""
    ).astype(np.int8)

    df[
        "country_missing_2"
    ] = (
        country2 == ""
    ).astype(np.int8)

    return df


# =====================================================================
# BUILD COMPLETE MODEL FEATURE MATRIX
# =====================================================================


def build_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the feature schema used by the saved LightGBM model."""

    df = prepare_batch(
        df
    )

    # ---------------------------------------------------------------
    # Text arrays
    # ---------------------------------------------------------------

    raw_name_1 = clean_text_series(
        df[
            "source1_business_name"
        ]
    )

    raw_name_2 = clean_text_series(
        df[
            "candidate_business_name"
        ]
    )

    name_norm_1 = clean_text_series(
        df[
            "source1_name_norm"
        ]
    )

    name_norm_2 = clean_text_series(
        df[
            "candidate_name_norm"
        ]
    )

    raw_address_1 = clean_text_series(
        df[
            "source1_business_address"
        ]
    )

    raw_address_2 = clean_text_series(
        df[
            "candidate_business_address"
        ]
    )

    address_norm_1 = clean_text_series(
        df[
            "source1_address_norm"
        ]
    )

    address_norm_2 = clean_text_series(
        df[
            "candidate_address_norm"
        ]
    )

    name_compare_1 = np.where(
        name_norm_1 != "",
        name_norm_1,
        raw_name_1,
    )

    name_compare_2 = np.where(
        name_norm_2 != "",
        name_norm_2,
        raw_name_2,
    )

    address_compare_1 = np.where(
        address_norm_1 != "",
        address_norm_1,
        raw_address_1,
    )

    address_compare_2 = np.where(
        address_norm_2 != "",
        address_norm_2,
        raw_address_2,
    )

    name_compact_1 = clean_text_series(
        df[
            "source1_name_compact"
        ]
    )

    name_compact_2 = clean_text_series(
        df[
            "candidate_name_compact"
        ]
    )

    address_compact_1 = clean_text_series(
        df[
            "source1_address_compact"
        ]
    )

    address_compact_2 = clean_text_series(
        df[
            "candidate_address_compact"
        ]
    )

    # ---------------------------------------------------------------
    # Feature dictionary
    # ---------------------------------------------------------------

    features: dict[str, Any] = {}

    # ===============================================================
    # NAME FEATURES
    # ===============================================================

    features[
        "name_exact"
    ] = (
        (
            raw_name_1 != ""
        )
        &
        (
            raw_name_2 != ""
        )
        &
        (
            raw_name_1
            ==
            raw_name_2
        )
    ).astype(np.int8)

    features[
        "name_norm_exact"
    ] = (
        (
            name_norm_1 != ""
        )
        &
        (
            name_norm_2 != ""
        )
        &
        (
            name_norm_1
            ==
            name_norm_2
        )
    ).astype(np.int8)

    features[
        "name_compact_exact"
    ] = (
        (
            name_compact_1 != ""
        )
        &
        (
            name_compact_2 != ""
        )
        &
        (
            name_compact_1
            ==
            name_compact_2
        )
    ).astype(np.int8)

    features[
        "name_ratio"
    ] = rapid_ratio(
        name_compare_1,
        name_compare_2,
    )

    features[
        "name_partial_ratio"
    ] = rapid_partial_ratio(
        name_compare_1,
        name_compare_2,
    )

    features[
        "name_jaro_winkler"
    ] = rapid_jaro_winkler(
        name_compare_1,
        name_compare_2,
    )

    features[
        "name_token_sort_ratio"
    ] = rapid_token_sort_ratio(
        name_compare_1,
        name_compare_2,
    )

    features[
        "name_token_set_ratio"
    ] = rapid_token_set_ratio(
        name_compare_1,
        name_compare_2,
    )

    features[
        "name_token_jaccard"
    ] = df[
        "name_lexical_token_jaccard"
    ].to_numpy(
        dtype=np.float32
    )

    features[
        "name_length_1"
    ] = df[
        "name_length_1"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_length_2"
    ] = df[
        "name_length_2"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_length_diff"
    ] = df[
        "name_length_diff"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_compact_length_1"
    ] = df[
        "name_compact_length_1"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_compact_length_2"
    ] = df[
        "name_compact_length_2"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_compact_length_diff"
    ] = df[
        "name_compact_length_diff"
    ].to_numpy(
        dtype=np.int32
    )

    # ===============================================================
    # ADDRESS FEATURES
    # ===============================================================

    features[
        "address_exact"
    ] = (
        (
            raw_address_1 != ""
        )
        &
        (
            raw_address_2 != ""
        )
        &
        (
            raw_address_1
            ==
            raw_address_2
        )
    ).astype(np.int8)

    features[
        "address_norm_exact"
    ] = (
        (
            address_norm_1 != ""
        )
        &
        (
            address_norm_2 != ""
        )
        &
        (
            address_norm_1
            ==
            address_norm_2
        )
    ).astype(np.int8)

    features[
        "address_compact_exact"
    ] = (
        (
            address_compact_1 != ""
        )
        &
        (
            address_compact_2 != ""
        )
        &
        (
            address_compact_1
            ==
            address_compact_2
        )
    ).astype(np.int8)

    features[
        "address_missing_1"
    ] = df[
        "address_missing_1"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "address_missing_2"
    ] = df[
        "address_missing_2"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "address_both_missing"
    ] = df[
        "address_both_missing"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "address_ratio"
    ] = rapid_ratio(
        address_compare_1,
        address_compare_2,
    )

    features[
        "address_partial_ratio"
    ] = rapid_partial_ratio(
        address_compare_1,
        address_compare_2,
    )

    features[
        "address_token_sort_ratio"
    ] = rapid_token_sort_ratio(
        address_compare_1,
        address_compare_2,
    )

    features[
        "address_token_set_ratio"
    ] = rapid_token_set_ratio(
        address_compare_1,
        address_compare_2,
    )

    features[
        "address_token_jaccard"
    ] = df[
        "address_lexical_token_jaccard"
    ].to_numpy(
        dtype=np.float32
    )

    features[
        "address_length_1"
    ] = df[
        "address_length_1"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "address_length_2"
    ] = df[
        "address_length_2"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "address_length_diff"
    ] = df[
        "address_length_diff"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "address_compact_length_1"
    ] = df[
        "address_compact_length_1"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "address_compact_length_2"
    ] = df[
        "address_compact_length_2"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "address_compact_length_diff"
    ] = df[
        "address_compact_length_diff"
    ].to_numpy(
        dtype=np.int32
    )

    # ===============================================================
    # NUMERIC FEATURES
    # ===============================================================

    features[
        "numeric_count_1"
    ] = df[
        "numeric_count_1"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "numeric_count_2"
    ] = df[
        "numeric_count_2"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "numeric_overlap_count"
    ] = df[
        "numeric_overlap_count"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "numeric_jaccard"
    ] = df[
        "numeric_jaccard"
    ].to_numpy(
        dtype=np.float32
    )

    features[
        "numeric_exact_match"
    ] = df[
        "numeric_exact_match"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "numeric_any_overlap"
    ] = df[
        "numeric_any_overlap"
    ].to_numpy(
        dtype=np.int8
    )

    # ===============================================================
    # COUNTRY FEATURES
    # ===============================================================

    features[
        "country_exact"
    ] = df[
        "country_exact"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "country_norm_exact"
    ] = df[
        "country_norm_exact"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "country_code_exact"
    ] = df[
        "country_code_exact"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "country_both_present"
    ] = df[
        "country_both_present"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "country_missing_1"
    ] = df[
        "country_missing_1"
    ].to_numpy(
        dtype=np.int8
    )

    features[
        "country_missing_2"
    ] = df[
        "country_missing_2"
    ].to_numpy(
        dtype=np.int8
    )

    # ===============================================================
    # NAME LEXICAL FEATURES
    # ===============================================================

    features[
        "name_lexical_exact"
    ] = (
        (
            name_norm_1 != ""
        )
        &
        (
            name_norm_2 != ""
        )
        &
        (
            name_norm_1
            ==
            name_norm_2
        )
    ).astype(np.int8)

    features[
        "name_lexical_ratio"
    ] = rapid_ratio(
        name_norm_1,
        name_norm_2,
    )

    features[
        "name_lexical_partial_ratio"
    ] = rapid_partial_ratio(
        name_norm_1,
        name_norm_2,
    )

    features[
        "name_lexical_token_sort_ratio"
    ] = rapid_token_sort_ratio(
        name_norm_1,
        name_norm_2,
    )

    features[
        "name_lexical_token_set_ratio"
    ] = rapid_token_set_ratio(
        name_norm_1,
        name_norm_2,
    )

    features[
        "name_lexical_token_jaccard"
    ] = df[
        "name_lexical_token_jaccard"
    ].to_numpy(
        dtype=np.float32
    )

    features[
        "name_lexical_token_count_1"
    ] = df[
        "name_lexical_token_count_1"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "name_lexical_token_count_2"
    ] = df[
        "name_lexical_token_count_2"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "name_lexical_token_overlap_count"
    ] = df[
        "name_lexical_token_overlap_count"
    ].to_numpy(
        dtype=np.int16
    )

    features[
        "name_lexical_length_1"
    ] = df[
        "name_lexical_length_1"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_lexical_length_2"
    ] = df[
        "name_lexical_length_2"
    ].to_numpy(
        dtype=np.int32
    )

    features[
        "name_lexical_length_diff"
    ] = df[
        "name_lexical_length_diff"
    ].to_numpy(
        dtype=np.int32
    )

    # ===============================================================
    # ADDRESS LEXICAL FEATURES
    #
    # IMPORTANT:
    #
    # pair_features() does:
    #
    #     f"address_lexical_{key}"
    #
    # while lexical_features() itself returns keys such as:
    #
    #     lexical_exact
    #     lexical_ratio
    #
    # Therefore the actual trained model has:
    #
    #     address_lexical_lexical_exact
    #     address_lexical_lexical_ratio
    #     ...
    #
    # ===============================================================

    address_lexical_exact = (
        (
            address_norm_1 != ""
        )
        &
        (
            address_norm_2 != ""
        )
        &
        (
            address_norm_1
            ==
            address_norm_2
        )
    ).astype(np.int8)

    address_lexical_ratio = (
        rapid_ratio(
            address_norm_1,
            address_norm_2,
        )
    )

    address_lexical_partial_ratio = (
        rapid_partial_ratio(
            address_norm_1,
            address_norm_2,
        )
    )

    address_lexical_token_sort_ratio = (
        rapid_token_sort_ratio(
            address_norm_1,
            address_norm_2,
        )
    )

    address_lexical_token_set_ratio = (
        rapid_token_set_ratio(
            address_norm_1,
            address_norm_2,
        )
    )

    address_lexical_token_jaccard = (
        df[
            "address_lexical_token_jaccard"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    address_lexical_token_count_1 = (
        df[
            "address_lexical_token_count_1"
        ]
        .to_numpy(
            dtype=np.int16
        )
    )

    address_lexical_token_count_2 = (
        df[
            "address_lexical_token_count_2"
        ]
        .to_numpy(
            dtype=np.int16
        )
    )

    address_lexical_token_overlap_count = (
        df[
            "address_lexical_token_overlap_count"
        ]
        .to_numpy(
            dtype=np.int16
        )
    )

    address_lexical_length_1 = (
        df[
            "address_lexical_length_1"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    address_lexical_length_2 = (
        df[
            "address_lexical_length_2"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    address_lexical_length_diff = (
        df[
            "address_lexical_length_diff"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    # Exact model-time names.
    features[
        "address_lexical_lexical_exact"
    ] = address_lexical_exact

    features[
        "address_lexical_lexical_ratio"
    ] = address_lexical_ratio

    features[
        "address_lexical_lexical_partial_ratio"
    ] = address_lexical_partial_ratio

    features[
        "address_lexical_lexical_token_sort_ratio"
    ] = address_lexical_token_sort_ratio

    features[
        "address_lexical_lexical_token_set_ratio"
    ] = address_lexical_token_set_ratio

    features[
        "address_lexical_lexical_token_jaccard"
    ] = address_lexical_token_jaccard

    features[
        "address_lexical_lexical_token_count_1"
    ] = address_lexical_token_count_1

    features[
        "address_lexical_lexical_token_count_2"
    ] = address_lexical_token_count_2

    features[
        "address_lexical_lexical_token_overlap_count"
    ] = address_lexical_token_overlap_count

    features[
        "address_lexical_lexical_length_1"
    ] = address_lexical_length_1

    features[
        "address_lexical_lexical_length_2"
    ] = address_lexical_length_2

    features[
        "address_lexical_lexical_length_diff"
    ] = address_lexical_length_diff

    # ===============================================================
    # CANDIDATE SOURCE
    # ===============================================================

    candidate_source = clean_text_series(
        df[
            "candidate_source"
        ]
    )

    features[
        "candidate_is_s2"
    ] = (
        candidate_source == "S2"
    ).astype(np.int8)

    features[
        "candidate_is_s3"
    ] = (
        candidate_source == "S3"
    ).astype(np.int8)

    result = pd.DataFrame(
        features
    )

    return result


# =====================================================================
# ALIGN TO SAVED MODEL
# =====================================================================


def align_model_features(
    features: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Ensure inference exactly matches training schema."""

    missing = [
        name
        for name in feature_names
        if name not in features.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing model features:\n"
            + "\n".join(
                f"  - {name}"
                for name in missing
            )
            + "\n\nAvailable features:\n"
            + "\n".join(
                f"  - {name}"
                for name in features.columns
            )
        )

    X = features[
        feature_names
    ].copy()

    if X.isnull().any().any():
        null_counts = (
            X.isnull()
            .sum()
            .loc[
                lambda s: s > 0
            ]
        )

        raise RuntimeError(
            "NaN values found in model features:\n"
            f"{null_counts.to_dict()}"
        )

    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric:
        raise RuntimeError(
            "Non-numeric features found:\n"
            f"{non_numeric}"
        )

    return X


# =====================================================================
# CREATE TOP-K CANDIDATES
# =====================================================================


def create_topk_candidates(
    con: duckdb.DuckDBPyConnection,
) -> int:
    """Create exactly the top-K candidate set used for inference."""

    print(
        "\nCreating top-K candidates..."
    )

    print(
        f"  K = {TOP_K}"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE s1_for_topk AS

        SELECT
            entity_id,
            country,
            name_compact,
            name_tokens,
            address_compact,
            address_tokens,
            address_numbers

        FROM s1;
        """
    )

    print(
        "\nScoring blocking candidates..."
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE scored_candidates AS

        SELECT

            c.source1_entity_id,
            c.candidate_entity_id,

            (
                CASE
                    WHEN
                        s1.name_compact <> ''
                        AND
                        s1.name_compact =
                        t.name_compact
                    THEN 1000.0
                    ELSE 0.0
                END

                +

                CASE
                    WHEN
                        s1.address_compact <> ''
                        AND
                        s1.address_compact =
                        t.address_compact
                    THEN 1000.0
                    ELSE 0.0
                END

                +

                CASE
                    WHEN
                        LOWER(
                            TRIM(
                                s1.country
                            )
                        ) <> ''
                        AND
                        LOWER(
                            TRIM(
                                s1.country
                            )
                        )
                        =
                        LOWER(
                            TRIM(
                                t.country
                            )
                        )
                    THEN 100.0
                    ELSE 0.0
                END

                +

                CASE
                    WHEN
                        list_unique(
                            list_concat(
                                COALESCE(
                                    s1.address_numbers,
                                    []
                                ),
                                COALESCE(
                                    t.address_numbers,
                                    []
                                )
                            )
                        ) = 0
                    THEN 0.0

                    ELSE

                        50.0
                        *
                        (
                            CAST(
                                list_unique(
                                    list_intersect(
                                        COALESCE(
                                            s1.address_numbers,
                                            []
                                        ),
                                        COALESCE(
                                            t.address_numbers,
                                            []
                                        )
                                    )
                                )
                                AS DOUBLE
                            )
                            /
                            list_unique(
                                list_concat(
                                    COALESCE(
                                        s1.address_numbers,
                                        []
                                    ),
                                    COALESCE(
                                        t.address_numbers,
                                        []
                                    )
                                )
                            )
                        )
                END

                +

                CASE
                    WHEN
                        list_unique(
                            list_concat(
                                COALESCE(
                                    s1.name_tokens,
                                    []
                                ),
                                COALESCE(
                                    t.name_tokens,
                                    []
                                )
                            )
                        ) = 0
                    THEN 0.0

                    ELSE

                        30.0
                        *
                        (
                            CAST(
                                list_unique(
                                    list_intersect(
                                        COALESCE(
                                            s1.name_tokens,
                                            []
                                        ),
                                        COALESCE(
                                            t.name_tokens,
                                            []
                                        )
                                    )
                                )
                                AS DOUBLE
                            )
                            /
                            list_unique(
                                list_concat(
                                    COALESCE(
                                        s1.name_tokens,
                                        []
                                    ),
                                    COALESCE(
                                        t.name_tokens,
                                        []
                                    )
                                )
                            )
                        )
                END

                +

                CASE
                    WHEN
                        list_unique(
                            list_concat(
                                COALESCE(
                                    s1.address_tokens,
                                    []
                                ),
                                COALESCE(
                                    t.address_tokens,
                                    []
                                )
                            )
                        ) = 0
                    THEN 0.0

                    ELSE

                        30.0
                        *
                        (
                            CAST(
                                list_unique(
                                    list_intersect(
                                        COALESCE(
                                            s1.address_tokens,
                                            []
                                        ),
                                        COALESCE(
                                            t.address_tokens,
                                            []
                                        )
                                    )
                                )
                                AS DOUBLE
                            )
                            /
                            list_unique(
                                list_concat(
                                    COALESCE(
                                        s1.address_tokens,
                                        []
                                    ),
                                    COALESCE(
                                        t.address_tokens,
                                        []
                                    )
                                )
                            )
                        )
                END

            ) AS heuristic_score

        FROM candidate_pairs c

        INNER JOIN s1_for_topk s1
            ON c.source1_entity_id =
               s1.entity_id

        INNER JOIN targets t
            ON c.candidate_entity_id =
               t.candidate_entity_id;
        """
    )

    print(
        "\nSelecting top candidates..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE topk_candidates AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM scored_candidates

        QUALIFY ROW_NUMBER()
        OVER (
            PARTITION BY
                source1_entity_id

            ORDER BY
                heuristic_score DESC,
                candidate_entity_id
        )
        <= {TOP_K};
        """
    )

    total = con.execute(
        """
        SELECT COUNT(*)
        FROM topk_candidates;
        """
    ).fetchone()[0]

    zero_candidates = con.execute(
        """
        SELECT COUNT(*)

        FROM s1

        LEFT JOIN topk_candidates c
            ON s1.entity_id =
               c.source1_entity_id

        WHERE c.source1_entity_id IS NULL;
        """
    ).fetchone()[0]

    print(
        f"Final candidate pairs: "
        f"{total:,}"
    )

    print(
        f"Zero-candidate S1: "
        f"{zero_candidates:,}"
    )

    return int(total)


# =====================================================================
# BATCH CREATION
# =====================================================================


def create_s1_batches(
    con: duckdb.DuckDBPyConnection,
) -> int:
    """Create Source-1 inference batches."""

    print(
        "\nCreating S1 inference batches..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE s1_batches AS

        SELECT
            entity_id
                AS source1_entity_id,

            CAST(
                FLOOR(
                    (
                        ROW_NUMBER()
                        OVER (
                            ORDER BY entity_id
                        )
                        - 1
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
        CREATE OR REPLACE TABLE topk_batched AS

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            b.batch_id

        FROM topk_candidates c

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


# =====================================================================
# FETCH ONE BATCH
# =====================================================================


def fetch_inference_batch(
    con: duckdb.DuckDBPyConnection,
    batch_id: int,
) -> tuple[
    list[str],
    pd.DataFrame,
]:
    """Fetch candidates for one group of S1 entities."""

    s1_rows = con.execute(
        """
        SELECT
            source1_entity_id

        FROM s1_batches

        WHERE batch_id = ?

        ORDER BY source1_entity_id;
        """,
        [batch_id],
    ).fetchall()

    batch_s1_ids = [
        str(row[0])
        for row in s1_rows
    ]

    # ---------------------------------------------------------------
    # Fetch only this batch.
    # ---------------------------------------------------------------

    df = con.execute(
        """
        SELECT

            c.source1_entity_id,
            c.candidate_entity_id,

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

        FROM topk_batched c

        INNER JOIN s1
            ON c.source1_entity_id =
               s1.entity_id

        INNER JOIN targets t
            ON c.candidate_entity_id =
               t.candidate_entity_id

        WHERE c.batch_id = ?

        ORDER BY
            c.source1_entity_id,
            c.candidate_entity_id;
        """,
        [batch_id],
    ).fetch_df()

    return (
        batch_s1_ids,
        df,
    )


# =====================================================================
# OUTPUT FILE CREATION
# =====================================================================


def create_output_files() -> tuple[
    Any,
    Any,
    csv.writer,
    csv.writer,
]:
    """Create fresh output files."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_handle = (
        CANDIDATE_OUTPUT.open(
            "w",
            encoding="utf-8",
            newline="",
        )
    )

    matching_handle = (
        MATCHING_OUTPUT.open(
            "w",
            encoding="utf-8",
            newline="",
        )
    )

    candidate_writer = csv.writer(
        candidate_handle,
        delimiter="\t",
        lineterminator="\n",
    )

    matching_writer = csv.writer(
        matching_handle,
        delimiter="\t",
        lineterminator="\n",
    )

    candidate_writer.writerow(
        [
            "source1_entity_id",
            "candidate_entity_ids",
        ]
    )

    matching_writer.writerow(
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    )

    return (
        candidate_handle,
        matching_handle,
        candidate_writer,
        matching_writer,
    )


# =====================================================================
# FINAL OUTPUT CONSISTENCY
# =====================================================================


def validate_match_subset(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Verify every selected match is a member of the TOP-K candidate set.

    This is checked again after inference, before submission.
    """

    print(
        "\nChecking match ⊆ candidate..."
    )

    # Read the matching file through DuckDB.
    # We only use this as a sanity check.
    matching_path = str(
        MATCHING_OUTPUT
    )

    invalid_count = con.execute(
        f"""
        WITH matching_rows AS (

            SELECT
                source1_entity_id,
                matched_entity_ids

            FROM read_csv(
                '{matching_path}',
                delim='\\t',
                header=true,
                quote='',
                escape=''
            )

            WHERE
                matched_entity_ids IS NOT NULL
                AND
                matched_entity_ids <> ''
        ),

        exploded AS (

            SELECT
                source1_entity_id,
                UNNEST(
                    string_split(
                        matched_entity_ids,
                        ','
                    )
                ) AS candidate_entity_id

            FROM matching_rows
        )

        SELECT COUNT(*)

        FROM exploded e

        LEFT JOIN topk_candidates c

            ON e.source1_entity_id =
               c.source1_entity_id

            AND
            e.candidate_entity_id =
               c.candidate_entity_id

        WHERE
            c.candidate_entity_id IS NULL;
        """
    ).fetchone()[0]

    if invalid_count != 0:
        raise RuntimeError(
            "Found "
            f"{invalid_count:,}"
            " matched IDs that are not present "
            "in the final candidate set."
        )

    print(
        "Match subset check: PASS"
    )


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:

    print("=" * 72)
    print(
        "FAST SUBMISSION #2 INFERENCE"
    )
    print("=" * 72)

    print(
        f"K: {TOP_K}"
    )

    print(
        f"Threshold: {THRESHOLD:.4f}"
    )

    print(
        f"DuckDB threads: "
        f"{DUCKDB_THREADS}"
    )

    print(
        "RapidFuzz workers: all CPU cores"
    )

    # ---------------------------------------------------------------
    # Required files
    # ---------------------------------------------------------------

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found:\n{DB_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    print(
        "\nLoading LightGBM model..."
    )

    model = LightGBMMatcher.load(
        MODEL_PATH
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
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
    # DuckDB
    # ---------------------------------------------------------------

    print(
        "\nOpening existing DuckDB..."
    )

    con = duckdb.connect(
        str(DB_PATH)
    )

    configure_duckdb(
        con
    )

    # ---------------------------------------------------------------
    # Verify original candidates
    # ---------------------------------------------------------------

    original_count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs;
        """
    ).fetchone()[0]

    print(
        f"Existing high-recall candidates: "
        f"{original_count:,}"
    )

    # ---------------------------------------------------------------
    # TOP-K
    # ---------------------------------------------------------------

    final_candidate_count = (
        create_topk_candidates(
            con
        )
    )

    print(
        f"\nFinal candidates to score: "
        f"{final_candidate_count:,}"
    )

    # ---------------------------------------------------------------
    # Batches
    # ---------------------------------------------------------------

    batch_count = create_s1_batches(
        con
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    (
        candidate_handle,
        matching_handle,
        candidate_writer,
        matching_writer,
    ) = create_output_files()

    total_scored = 0
    total_selected = 0
    total_nonempty_matches = 0

    try:

        print(
            "\nStarting batched feature/scoring pass..."
        )

        for batch_id in range(
            batch_count
        ):

            (
                batch_s1_ids,
                batch_df,
            ) = fetch_inference_batch(
                con,
                batch_id,
            )

            # Output dictionaries for every S1 in batch.
            candidate_groups: dict[
                str,
                list[str],
            ] = {
                source1_id: []
                for source1_id
                in batch_s1_ids
            }

            match_groups: dict[
                str,
                list[str],
            ] = {
                source1_id: []
                for source1_id
                in batch_s1_ids
            }

            if not batch_df.empty:

                # ---------------------------------------------------
                # Vectorized features
                # ---------------------------------------------------

                model_features = (
                    build_model_features(
                        batch_df
                    )
                )

                # ---------------------------------------------------
                # Schema alignment
                # ---------------------------------------------------

                X = align_model_features(
                    model_features,
                    feature_names,
                )

                # ---------------------------------------------------
                # Prediction
                # ---------------------------------------------------

                probabilities = (
                    model.predict_proba(
                        X
                    )
                )

                batch_df[
                    "probability"
                ] = probabilities

                # ---------------------------------------------------
                # Group candidates
                # ---------------------------------------------------

                for source1_id, group in (
                    batch_df.groupby(
                        "source1_entity_id",
                        sort=False,
                    )
                ):

                    source1_id = str(
                        source1_id
                    )

                    candidates = (
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
                                "probability"
                            ]
                            >= THRESHOLD
                        ][
                            "candidate_entity_id"
                        ]
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )

                    candidate_groups[
                        source1_id
                    ].extend(
                        candidates
                    )

                    match_groups[
                        source1_id
                    ].extend(
                        matches
                    )

                total_scored += len(
                    batch_df
                )

                total_selected += int(
                    (
                        probabilities
                        >= THRESHOLD
                    ).sum()
                )

            # -------------------------------------------------------
            # Write all S1 entities in this batch.
            # -------------------------------------------------------

            for source1_id in batch_s1_ids:

                candidates = sorted(
                    set(
                        candidate_groups[
                            source1_id
                        ]
                    )
                )

                matches = sorted(
                    set(
                        match_groups[
                            source1_id
                        ]
                    )
                )

                candidate_writer.writerow(
                    [
                        source1_id,
                        ",".join(
                            candidates
                        ),
                    ]
                )

                matching_writer.writerow(
                    [
                        source1_id,
                        ",".join(
                            matches
                        ),
                    ]
                )

                if matches:
                    total_nonempty_matches += 1

            # -------------------------------------------------------
            # Progress
            # -------------------------------------------------------

            if (
                (
                    batch_id + 1
                )
                % 5
                == 0
                or
                (
                    batch_id + 1
                )
                == batch_count
            ):

                print(
                    f"  batch "
                    f"{batch_id + 1:,}/"
                    f"{batch_count:,} | "
                    f"scored="
                    f"{total_scored:,} | "
                    f"selected="
                    f"{total_selected:,}"
                )

    finally:

        candidate_handle.close()
        matching_handle.close()

    # ---------------------------------------------------------------
    # Consistency check
    # ---------------------------------------------------------------

    validate_match_subset(
        con
    )

    # ---------------------------------------------------------------
    # Close DB
    # ---------------------------------------------------------------

    con.close()

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "FAST SUBMISSION #2 INFERENCE COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Final candidate pairs: "
        f"{final_candidate_count:,}"
    )

    print(
        f"Candidate pairs scored: "
        f"{total_scored:,}"
    )

    print(
        f"Selected matches: "
        f"{total_selected:,}"
    )

    print(
        f"S1 entities with >=1 match: "
        f"{total_nonempty_matches:,}"
    )

    print(
        f"\nCandidate file:"
        f"\n  {CANDIDATE_OUTPUT}"
    )

    print(
        f"\nMatching file:"
        f"\n  {MATCHING_OUTPUT}"
    )

    print(
        "\nRun the official validator next."
    )


if __name__ == "__main__":
    main()