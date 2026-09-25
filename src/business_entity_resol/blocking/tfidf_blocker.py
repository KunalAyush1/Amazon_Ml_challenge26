"""TF-IDF based candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfCandidate:
    """Candidate returned by the TF-IDF blocker."""

    candidate_id: str
    source: str
    rank: int
    similarity: float


class TfidfBlocker:
    """Generate candidates using character-level TF-IDF similarity."""

    def __init__(
        self,
        *,
        fields: tuple[str, ...] = ("name_norm", "address_norm"),
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 5),
        top_k: int = 50,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must be a valid positive range.")

        self.fields = fields
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.top_k = top_k

        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._candidate_ids: list[str] = []
        self._candidate_sources: list[str] = []

    def fit(
        self,
        source2_records: list[dict[str, Any]],
        source3_records: list[dict[str, Any]],
    ) -> "TfidfBlocker":
        """Fit TF-IDF on Source-2 and Source-3 candidate records."""
        records: list[tuple[dict[str, Any], str]] = [
            (record, "S2") for record in source2_records
        ] + [
            (record, "S3") for record in source3_records
        ]

        self._candidate_ids = []
        self._candidate_sources = []

        documents: list[str] = []

        for record, source in records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                continue

            document = self._build_document(record)

            # Ignore records with no usable text.
            if not document:
                continue

            self._candidate_ids.append(str(entity_id))
            self._candidate_sources.append(source)
            documents.append(document)

        if not documents:
            self._vectorizer = None
            self._matrix = None
            return self

        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
        )

        self._matrix = self._vectorizer.fit_transform(documents)

        return self

    def retrieve(
        self,
        source1_record: dict[str, Any],
    ) -> list[TfidfCandidate]:
        """Retrieve the top-k TF-IDF candidates for one Source-1 record."""
        if self._vectorizer is None or self._matrix is None:
            return []

        document = self._build_document(source1_record)

        if not document:
            return []

        query_vector = self._vectorizer.transform([document])

        # TF-IDF vectors are L2-normalized by default, so the dot
        # product is cosine similarity.
        similarities = (self._matrix @ query_vector.T).toarray().ravel()

        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda index: (
                -float(similarities[index]),
                self._candidate_ids[index],
            ),
        )

        results: list[TfidfCandidate] = []

        for index in ranked_indices[: self.top_k]:
            similarity = float(similarities[index])

            # Do not return candidates with zero textual similarity.
            if similarity <= 0.0:
                continue

            results.append(
                TfidfCandidate(
                    candidate_id=self._candidate_ids[index],
                    source=self._candidate_sources[index],
                    rank=len(results) + 1,
                    similarity=similarity,
                )
            )

        return results

    def retrieve_many(
        self,
        source1_records: list[dict[str, Any]],
    ) -> dict[str, list[TfidfCandidate]]:
        """Retrieve TF-IDF candidates for multiple Source-1 records."""
        results: dict[str, list[TfidfCandidate]] = {}

        for record in source1_records:
            entity_id = record.get("entity_id")

            if entity_id is None:
                continue

            results[str(entity_id)] = self.retrieve(record)

        return results

    def _build_document(self, record: dict[str, Any]) -> str:
        """Build one searchable document from configured fields."""
        values: list[str] = []

        for field in self.fields:
            value = record.get(field)

            if value is None:
                continue

            value = str(value).strip()

            if value:
                values.append(value)

        return " ".join(values)