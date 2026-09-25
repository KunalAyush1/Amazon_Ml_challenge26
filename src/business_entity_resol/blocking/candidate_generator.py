from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .candidate_union import CandidateUnion
from .dense_blocker import DenseBlocker
from .exact_blocker import ExactBlocker
from .numeric_blocker import NumericBlocker
from .rare_token_blocker import RareTokenBlocker
from .tfidf_blocker import TfidfBlocker


class CandidateGenerator:
    """
    Generate and merge S1 -> S2/S3 candidate pairs.

    The generator assumes records have already been normalized by the
    preprocessing layer.
    """

    def __init__(
        self,
        *,
        exact_blocker: ExactBlocker | None = None,
        rare_token_blocker: RareTokenBlocker | None = None,
        numeric_blocker: NumericBlocker | None = None,
        tfidf_blocker: TfidfBlocker | None = None,
        dense_blocker: DenseBlocker | None = None,
    ) -> None:
        self.exact_blocker = exact_blocker or ExactBlocker()
        self.rare_token_blocker = (
            rare_token_blocker or RareTokenBlocker()
        )
        self.numeric_blocker = numeric_blocker or NumericBlocker()
        self.tfidf_blocker = tfidf_blocker or TfidfBlocker()
        self.dense_blocker = dense_blocker or DenseBlocker()

        self._fitted = False

    def fit(
        self,
        source2_records: Iterable[dict[str, Any]],
        source3_records: Iterable[dict[str, Any]],
    ) -> "CandidateGenerator":
        """
        Fit all blockers on Source-2 and Source-3 records.
        """

        source2_records = list(source2_records)
        source3_records = list(source3_records)

        self.exact_blocker.fit(
            source2_records,
            source3_records,
        )

        self.rare_token_blocker.fit(
            source2_records,
            source3_records,
        )

        self.numeric_blocker.fit(
            source2_records,
            source3_records,
        )

        self.tfidf_blocker.fit(
            source2_records,
            source3_records,
        )

        self.dense_blocker.fit(
            source2_records,
            source3_records,
        )

        self._fitted = True

        return self

    def generate(
        self,
        source1_records: Iterable[dict[str, Any]],
    ) -> CandidateUnion:
        """
        Generate the union of candidates from all blockers.
        """

        if not self._fitted:
            raise RuntimeError(
                "CandidateGenerator must be fitted before generate()."
            )

        source1_records = list(source1_records)

        union = CandidateUnion()

        self._add_exact_candidates(
            source1_records,
            union,
        )

        self._add_rare_token_candidates(
            source1_records,
            union,
        )

        self._add_numeric_candidates(
            source1_records,
            union,
        )

        self._add_tfidf_candidates(
            source1_records,
            union,
        )

        self._add_dense_candidates(
            source1_records,
            union,
        )

        return union

    def _add_exact_candidates(
        self,
        source1_records: list[dict[str, Any]],
        union: CandidateUnion,
    ) -> None:
        results = self.exact_blocker.retrieve_many(
            source1_records,
        )

        for source1_entity_id, candidates in results.items():
            for candidate in candidates:
                union.add(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=candidate.candidate_id,
                    source=candidate.source,
                    blocker="exact",
                )

    def _add_rare_token_candidates(
        self,
        source1_records: list[dict[str, Any]],
        union: CandidateUnion,
    ) -> None:
        results = self.rare_token_blocker.retrieve_many(
            source1_records,
        )

        for source1_entity_id, candidates in results.items():
            for rank, candidate in enumerate(candidates, start=1):
                union.add(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=candidate.candidate_id,
                    source=candidate.source,
                    blocker="rare_token",
                    rank=rank,
                )

    def _add_numeric_candidates(
        self,
        source1_records: list[dict[str, Any]],
        union: CandidateUnion,
    ) -> None:
        results = self.numeric_blocker.retrieve_many(
            source1_records,
        )

        for source1_entity_id, candidates in results.items():
            for rank, candidate in enumerate(candidates, start=1):
                union.add(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=candidate.candidate_id,
                    source=candidate.source,
                    blocker="numeric",
                    rank=rank,
                )

    def _add_tfidf_candidates(
        self,
        source1_records: list[dict[str, Any]],
        union: CandidateUnion,
    ) -> None:
        results = self.tfidf_blocker.retrieve_many(
            source1_records,
        )

        for source1_entity_id, candidates in results.items():
            for candidate in candidates:
                union.add(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=candidate.candidate_id,
                    source=candidate.source,
                    blocker="tfidf",
                    rank=candidate.rank,
                )

    def _add_dense_candidates(
        self,
        source1_records: list[dict[str, Any]],
        union: CandidateUnion,
    ) -> None:
        results = self.dense_blocker.retrieve_many(
            source1_records,
        )

        for source1_entity_id, candidates in results.items():
            for rank, candidate_id in enumerate(candidates, start=1):
                candidate = self.dense_blocker.get_candidate_record(
                    candidate_id
                )

                union.add(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=candidate_id,
                    source=candidate["_source"],
                    blocker="dense",
                    rank=rank,
                )