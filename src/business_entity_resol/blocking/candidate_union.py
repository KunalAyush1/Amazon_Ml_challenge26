from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .candidate_store import CandidateRecord


class CandidateUnion:
    """
    Merge candidate results from multiple blocking methods.

    Candidates are deduplicated by:

        (source1_entity_id, candidate_entity_id)

    while preserving blocker provenance and retrieval ranks.
    """

    def __init__(self) -> None:
        self._candidates: dict[
            tuple[str, str],
            CandidateRecord,
        ] = {}

    def add(
        self,
        source1_entity_id: str,
        candidate_entity_id: str,
        source: str,
        blocker: str,
        rank: int | None = None,
    ) -> CandidateRecord:
        """Add one candidate and preserve its provenance."""

        source1_entity_id = str(source1_entity_id)
        candidate_entity_id = str(candidate_entity_id)
        source = str(source)
        blocker = str(blocker)

        key = (
            source1_entity_id,
            candidate_entity_id,
        )

        if key not in self._candidates:
            self._candidates[key] = CandidateRecord(
                source1_entity_id=source1_entity_id,
                candidate_entity_id=candidate_entity_id,
                source=source,
            )
        else:
            existing = self._candidates[key]

            if existing.source != source:
                raise ValueError(
                    "Candidate source changed for the same "
                    f"candidate pair: {key}"
                )

        candidate = self._candidates[key]

        candidate.add_provenance(
            blocker=blocker,
            rank=rank,
        )

        return candidate

    def add_many(
        self,
        candidates: Iterable[Mapping[str, Any]],
        blocker: str,
    ) -> None:
        """
        Add candidates from a generic iterable of dictionaries.

        Expected fields:

            source1_entity_id
            candidate_entity_id
            source

        Optional:

            rank
        """

        for candidate in candidates:
            self.add(
                source1_entity_id=candidate["source1_entity_id"],
                candidate_entity_id=candidate["candidate_entity_id"],
                source=candidate["source"],
                blocker=blocker,
                rank=candidate.get("rank"),
            )

    def get(
        self,
        source1_entity_id: str | None = None,
    ) -> list[CandidateRecord]:
        """
        Return candidates.

        If source1_entity_id is supplied, return only candidates
        belonging to that Source-1 entity.
        """

        candidates = list(self._candidates.values())

        if source1_entity_id is not None:
            source1_entity_id = str(source1_entity_id)

            candidates = [
                candidate
                for candidate in candidates
                if candidate.source1_entity_id == source1_entity_id
            ]

        candidates.sort(
            key=lambda candidate: (
                candidate.source1_entity_id,
                candidate.source,
                candidate.candidate_entity_id,
            )
        )

        return candidates

    def get_for_entity(
        self,
        source1_entity_id: str,
    ) -> list[CandidateRecord]:
        """Convenience wrapper for retrieving one S1 entity."""
        return self.get(source1_entity_id)

    def to_dicts(
        self,
        source1_entity_id: str | None = None,
    ) -> list[dict]:
        """Return candidates as serializable dictionaries."""
        return [
            candidate.to_dict()
            for candidate in self.get(source1_entity_id)
        ]

    def __len__(self) -> int:
        return len(self._candidates)

    def clear(self) -> None:
        """Remove all stored candidates."""
        self._candidates.clear()