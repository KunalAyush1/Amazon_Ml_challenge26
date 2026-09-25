from collections.abc import Mapping
from typing import Any

from business_entity_resol.preprocessing.address_normalizer import (
    normalize_address_record,
)
from business_entity_resol.preprocessing.name_normalizer import (
    normalize_name_record,
)


REQUIRED_RECORD_FIELDS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize one business record while preserving the original values.
    """
    missing_fields = [
        field
        for field in REQUIRED_RECORD_FIELDS
        if field not in record
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required record fields: {missing_fields}"
        )

    name_fields = normalize_name_record(
        record.get("business_name")
    )

    address_fields = normalize_address_record(
        record.get("business_address")
    )

    return {
        "entity_id": record.get("entity_id"),
        "business_name": record.get("business_name"),
        "business_address": record.get("business_address"),
        "country": record.get("country"),
        **name_fields,
        **address_fields,
    }


def normalize_dataframe(df):
    """
    Normalize every row of a source dataframe.

    Returns a new dataframe and does not modify the input dataframe.
    """
    records = [
        normalize_record(record)
        for record in df.to_dict(orient="records")
    ]

    return type(df)(records)