from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RareTokenCandidate:
    """Candidate returned by the rare-token blocking stage."""

    candidate_id: str
    source: str
    matched_tokens: tuple[str, ...]


class RareTokenBlocker:
    """
    Candidate generation using informative/rare tokens.

    Token frequencies are learned from Source-2 and Source-3.
    Source-1 records are queried against those indexes.

    Normalization/tokenization is expected to have already been
    performed by the preprocessing layer.
    """

    DEFAULT_FIELDS = (
        "name_tokens",
        "address_tokens",
    )

    def __init__(
        self,
        fields: Iterable[str] | None = None,
        max_token_frequency: int = 100,
        max_tokens_per_record: int = 5,
    ) -> None:
        if max_token_frequency < 1:
            raise ValueError("max_token_frequency must be >= 1")

        if max_tokens_per_record < 1:
            raise ValueError("max_tokens_per_record must be >= 1")

        self.fields = tuple(fields or self.DEFAULT_FIELDS)

        if not self.fields:
            raise ValueError("At least one token field is required")

        self.max_token_frequency = max_token_frequency
        self.max_tokens_per_record = max_tokens_per_record

        # field -> token -> number of records containing token
        self._frequencies: dict[str, Counter[str]] = {
            field: Counter() for field in self.fields
        }

        # field -> token -> candidate IDs
        self._indexes: dict[str, dict[str, set[str]]] = {
            field: defaultdict(set) for field in self.fields
        }

        # candidate ID -> source
        self._candidate_sources: dict[str, str] = {}

    @staticmethod
    def _clean_tokens(value: Any) -> list[str]:
        """Convert a token field into a clean list of tokens."""

        if value is None:
            return []

        if isinstance(value, str):
            value = value.split()

        tokens = []

        for token in value:
            if token is None:
                continue

            token = str(token).strip().lower()

            if token:
                tokens.append(token)

        return list(dict.fromkeys(tokens))

    def fit(
        self,
        source2_records: Iterable[dict[str, Any]],
        source3_records: Iterable[dict[str, Any]],
    ) -> "RareTokenBlocker":
        """Build token frequency tables and inverted indexes."""

        self._frequencies = {
            field: Counter() for field in self.fields
        }

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
                tokens = self._clean_tokens(record.get(field))

                # Count each token once per record.
                for token in set(tokens):
                    self._frequencies[field][token] += 1
                    self._indexes[field][token].add(entity_id)

    def token_frequency(
        self,
        field: str,
        token: str,
    ) -> int:
        """Return the number of target records containing a token."""

        if field not in self._frequencies:
            raise ValueError(f"Unknown field: {field}")

        return self._frequencies[field].get(
            token.lower(),
            0,
        )

    def _select_informative_tokens(
        self,
        field: str,
        tokens: list[str],
    ) -> list[str]:
        """
        Select the rarest usable tokens.

        Very common tokens are ignored. Among usable tokens,
        the least frequent tokens are preferred.
        """

        usable = [
            token
            for token in set(tokens)
            if 0 < self._frequencies[field].get(token, 0)
            <= self.max_token_frequency
        ]

        usable.sort(
            key=lambda token: (
                self._frequencies[field][token],
                token,
            )
        )

        return usable[: self.max_tokens_per_record]

    def retrieve(
        self,
        source1_record: dict[str, Any],
    ) -> list[RareTokenCandidate]:
        """Retrieve candidates using informative tokens."""

        candidates: dict[str, set[str]] = defaultdict(set)

        for field in self.fields:
            tokens = self._clean_tokens(source1_record.get(field))

            informative_tokens = self._select_informative_tokens(
                field,
                tokens,
            )

            for token in informative_tokens:
                candidate_ids = self._indexes[field].get(
                    token,
                    set(),
                )

                for candidate_id in candidate_ids:
                    candidates[candidate_id].add(token)

        result = [
            RareTokenCandidate(
                candidate_id=candidate_id,
                source=self._candidate_sources[candidate_id],
                matched_tokens=tuple(sorted(tokens)),
            )
            for candidate_id, tokens in candidates.items()
        ]

        result.sort(
            key=lambda candidate: (
                candidate.candidate_id,
            )
        )

        return result

    def retrieve_many(
        self,
        source1_records: Iterable[dict[str, Any]],
    ) -> dict[str, list[RareTokenCandidate]]:
        """Retrieve rare-token candidates for multiple S1 records."""

        results: dict[str, list[RareTokenCandidate]] = {}

        for record in source1_records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                raise ValueError(
                    "Every record must contain 'entity_id'"
                )

            entity_id = str(entity_id)

            results[entity_id] = self.retrieve(record)

        return results