from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExactCandidate:
    """Candidate returned by the exact blocking stage."""

    candidate_id: str
    source: str
    matched_keys: tuple[str, ...]


class ExactBlocker:
    """
    Exact blocking using normalized business fields.

    The blocker builds inverted indexes over target sources (S2/S3)
    and retrieves candidates for Source-1 records.

    No normalization is performed here. The blocker expects normalized
    fields to already be available in the input records.
    """

    DEFAULT_FIELDS = (
        "name_norm",
        "name_compact",
        "address_norm",
        "address_compact",
    )

    def __init__(
        self,
        fields: Iterable[str] | None = None,
        max_frequency: int = 100,
    ) -> None:
        if max_frequency < 1:
            raise ValueError("max_frequency must be >= 1")

        self.fields = tuple(fields or self.DEFAULT_FIELDS)
        if not self.fields:
            raise ValueError("At least one blocking field is required")

        self.max_frequency = max_frequency

        # field -> normalized value -> candidate IDs
        self._indexes: dict[str, dict[str, set[str]]] = {
            field: defaultdict(set) for field in self.fields
        }

        # candidate ID -> source
        self._candidate_sources: dict[str, str] = {}

    @staticmethod
    def _clean_value(value: Any) -> str | None:
        """Convert a field value into a usable blocking key."""
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value

    def fit(
        self,
        source2_records: Iterable[dict[str, Any]],
        source3_records: Iterable[dict[str, Any]],
    ) -> "ExactBlocker":
        """
        Build indexes from Source-2 and Source-3 records.

        Source-1 records are never indexed because candidates are only
        allowed to come from S2 and S3.
        """
        self._indexes = {
            field: defaultdict(set) for field in self.fields
        }
        self._candidate_sources = {}

        self._index_source(source2_records, "S2")
        self._index_source(source3_records, "S3")

        return self

    def _index_source(
        self,
        records: Iterable[dict[str, Any]],
        source: str,
    ) -> None:
        for record in records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                raise ValueError("Every record must contain 'entity_id'")

            entity_id = str(entity_id)

            if entity_id in self._candidate_sources:
                raise ValueError(
                    f"Duplicate candidate entity_id detected: {entity_id}"
                )

            self._candidate_sources[entity_id] = source

            for field in self.fields:
                value = self._clean_value(record.get(field))

                if value is None:
                    continue

                self._indexes[field][value].add(entity_id)

    def retrieve(
        self,
        source1_record: dict[str, Any],
    ) -> list[ExactCandidate]:
        """
        Retrieve exact-match candidates for one Source-1 record.

        A candidate is returned if at least one configured normalized
        field matches exactly.

        Very frequent keys are skipped according to max_frequency.
        """
        candidates: dict[str, set[str]] = defaultdict(set)

        for field in self.fields:
            value = self._clean_value(source1_record.get(field))

            if value is None:
                continue

            matching_ids = self._indexes[field].get(value, set())

            # Frequency cap: skip extremely common blocking values.
            if len(matching_ids) > self.max_frequency:
                continue

            for candidate_id in matching_ids:
                candidates[candidate_id].add(field)

        result = [
            ExactCandidate(
                candidate_id=candidate_id,
                source=self._candidate_sources[candidate_id],
                matched_keys=tuple(sorted(fields)),
            )
            for candidate_id, fields in candidates.items()
        ]

        result.sort(key=lambda candidate: candidate.candidate_id)

        return result

    def retrieve_many(
        self,
        source1_records: Iterable[dict[str, Any]],
    ) -> dict[str, list[ExactCandidate]]:
        """Retrieve exact candidates for multiple Source-1 records."""
        results: dict[str, list[ExactCandidate]] = {}

        for record in source1_records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                raise ValueError("Every record must contain 'entity_id'")

            entity_id = str(entity_id)

            results[entity_id] = self.retrieve(record)

        return results