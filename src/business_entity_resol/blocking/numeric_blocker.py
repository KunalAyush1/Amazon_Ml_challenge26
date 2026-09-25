from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NumericCandidate:
    """Candidate returned by numeric blocking."""

    candidate_id: str
    source: str
    matched_numbers: tuple[str, ...]


class NumericBlocker:
    """
    Candidate generation using exact overlap of extracted numbers.

    The numeric values are expected to be produced by the preprocessing
    layer. This blocker does not attempt to identify postal codes,
    house numbers, phone numbers, etc.
    """

    DEFAULT_FIELD = "address_numbers"

    def __init__(
        self,
        field: str = DEFAULT_FIELD,
        max_frequency: int = 100,
    ) -> None:
        if not field:
            raise ValueError("field must not be empty")

        if max_frequency < 1:
            raise ValueError("max_frequency must be >= 1")

        self.field = field
        self.max_frequency = max_frequency

        # number -> candidate IDs
        self._index: dict[str, set[str]] = defaultdict(set)

        # candidate ID -> source
        self._candidate_sources: dict[str, str] = {}

    @staticmethod
    def _clean_numbers(value: Any) -> list[str]:
        """Convert extracted numeric values into unique strings."""

        if value is None:
            return []

        if isinstance(value, str):
            value = value.split()

        numbers: list[str] = []

        for number in value:
            if number is None:
                continue

            number = str(number).strip()

            if number:
                numbers.append(number)

        return list(dict.fromkeys(numbers))

    def fit(
        self,
        source2_records: Iterable[dict[str, Any]],
        source3_records: Iterable[dict[str, Any]],
    ) -> "NumericBlocker":
        """Build the numeric inverted index."""

        self._index = defaultdict(set)
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

            numbers = self._clean_numbers(
                record.get(self.field)
            )

            for number in numbers:
                self._index[number].add(entity_id)

    def number_frequency(self, number: str) -> int:
        """Return how many target records contain a number."""

        return len(self._index.get(str(number), set()))

    def retrieve(
        self,
        source1_record: dict[str, Any],
    ) -> list[NumericCandidate]:
        """Retrieve candidates sharing at least one number."""

        candidates: dict[str, set[str]] = defaultdict(set)

        numbers = self._clean_numbers(
            source1_record.get(self.field)
        )

        for number in numbers:
            candidate_ids = self._index.get(number, set())

            # Skip extremely common numeric values.
            if len(candidate_ids) > self.max_frequency:
                continue

            for candidate_id in candidate_ids:
                candidates[candidate_id].add(number)

        result = [
            NumericCandidate(
                candidate_id=candidate_id,
                source=self._candidate_sources[candidate_id],
                matched_numbers=tuple(sorted(numbers)),
            )
            for candidate_id, numbers in candidates.items()
        ]

        # Correct the matched numbers to only those actually shared.
        result = [
            NumericCandidate(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                matched_numbers=tuple(
                    sorted(
                        set(candidate.matched_numbers)
                        & set(self._clean_numbers(
                            source1_record.get(self.field)
                        ))
                    )
                ),
            )
            for candidate in result
        ]

        result.sort(
            key=lambda candidate: candidate.candidate_id
        )

        return result

    def retrieve_many(
        self,
        source1_records: Iterable[dict[str, Any]],
    ) -> dict[str, list[NumericCandidate]]:
        """Retrieve numeric candidates for multiple S1 records."""

        results: dict[str, list[NumericCandidate]] = {}

        for record in source1_records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                raise ValueError(
                    "Every record must contain 'entity_id'"
                )

            entity_id = str(entity_id)

            results[entity_id] = self.retrieve(record)

        return results