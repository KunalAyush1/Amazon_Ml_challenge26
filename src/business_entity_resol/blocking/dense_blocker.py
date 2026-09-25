from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseBlocker:
    """
    Dense embedding based candidate retrieval blocker.

    Source-2 and Source-3 records are embedded into the same vector space.
    For each Source-1 record, the blocker retrieves the top-k most similar
    Source-2/Source-3 candidates.

    This blocker is used only for candidate generation.
    """

    DEFAULT_MODEL = "intfloat/e5-small-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 50,
        fields: Sequence[str] = (
            "name_norm",
            "address_norm",
            "country",
        ),
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.model_name = model_name
        self.top_k = top_k
        self.fields = tuple(fields)

        self.model: SentenceTransformer | None = None

        self._candidate_ids: list[str] = []
        self._candidate_records: list[dict[str, Any]] = []
        self._candidate_embeddings: np.ndarray | None = None

        self._fitted = False

    def _build_text(self, record: dict[str, Any]) -> str:
        """Build the text representation used for embedding."""

        parts: list[str] = []

        for field in self.fields:
            value = record.get(field, "")

            if value is None:
                value = ""

            value = str(value).strip()

            if value:
                parts.append(value)

        return " | ".join(parts)

    def fit(
        self,
        source2_records: Iterable[dict[str, Any]],
        source3_records: Iterable[dict[str, Any]],
    ) -> "DenseBlocker":
        """
        Fit the dense index using Source-2 and Source-3 records.
        """

        source2 = list(source2_records)
        source3 = list(source3_records)

        if not source2 and not source3:
            raise ValueError(
                "At least one candidate record is required"
            )

        # Preserve the source of every candidate so that CandidateUnion
        # can distinguish Source-2 from Source-3 records.
        records = [
            {**record, "_source": "S2"}
            for record in source2
        ] + [
            {**record, "_source": "S3"}
            for record in source3
        ]

        self.model = SentenceTransformer(self.model_name)

        texts = [
            self._build_text(record)
            for record in records
        ]

        self._candidate_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self._candidate_records = records

        self._candidate_ids = [
            str(record["entity_id"])
            for record in records
        ]

        self._fitted = True

        return self

    def retrieve(
        self,
        record: dict[str, Any],
    ) -> list[str]:
        """
        Retrieve the top-k most similar Source-2/Source-3 candidates.
        """

        if not self._fitted:
            raise RuntimeError(
                "DenseBlocker must be fitted before retrieve()"
            )

        if self.model is None:
            raise RuntimeError(
                "Dense blocker model is not initialized"
            )

        if self._candidate_embeddings is None:
            raise RuntimeError(
                "Dense blocker index is not initialized"
            )

        query_text = self._build_text(record)

        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        similarities = self._candidate_embeddings @ query_embedding

        k = min(
            self.top_k,
            len(self._candidate_ids),
        )

        # Sort by similarity descending.
        # Entity ID is used as a deterministic tie-breaker.
        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda index: (
                -float(similarities[index]),
                self._candidate_ids[index],
            ),
        )[:k]

        return [
            self._candidate_ids[index]
            for index in ranked_indices
        ]

    def retrieve_many(
        self,
        records: Iterable[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """
        Retrieve candidates for multiple Source-1 records.
        """

        return {
            str(record["entity_id"]): self.retrieve(record)
            for record in records
        }

    def get_candidate_record(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:
        """
        Return the indexed candidate record for an entity ID.
        """

        candidate_id = str(candidate_id)

        for record in self._candidate_records:
            if str(record["entity_id"]) == candidate_id:
                return record

        raise KeyError(
            f"Candidate entity not found: {candidate_id}"
        )

    def __len__(self) -> int:
        return len(self._candidate_ids)