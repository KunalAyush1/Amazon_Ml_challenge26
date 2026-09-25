"""Combined pairwise feature generation."""

from __future__ import annotations

from typing import Any

from .address_features import address_features
from .blocker_features import blocker_features
from .country_features import country_features
from .lexical_features import lexical_features
from .name_features import name_features
from .numeric_features import numeric_features


def pair_features(
    source1: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | int]:
    """
    Generate the complete feature vector for one candidate pair.

    Parameters
    ----------
    source1:
        Normalized Source-1 record.

    candidate:
        Normalized Source-2/Source-3 candidate record. Blocking
        provenance fields may also be present.

    Returns
    -------
    dict
        Combined name, address, numeric, country, lexical, and
        blocker-provenance features.
    """

    features: dict[str, float | int] = {}

    features.update(
        name_features(source1, candidate)
    )

    features.update(
        address_features(source1, candidate)
    )

    features.update(
        numeric_features(source1, candidate)
    )

    features.update(
        country_features(source1, candidate)
    )

    # General lexical features on normalized name.
    features.update(
        {
            f"name_{key}": value
            for key, value in lexical_features(
                source1,
                candidate,
                field="name_norm",
            ).items()
        }
    )

    # General lexical features on normalized address.
    features.update(
        {
            f"address_lexical_{key}": value
            for key, value in lexical_features(
                source1,
                candidate,
                field="address_norm",
            ).items()
        }
    )

    features.update(
        blocker_features(candidate)
    )

    return features