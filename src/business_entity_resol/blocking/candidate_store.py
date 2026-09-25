from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateRecord:
    """
    Represents one unique candidate for a Source-1 entity.

    A candidate may be discovered by multiple blocking methods.
    """

    source1_entity_id: str
    candidate_entity_id: str
    source: str

    blocker_provenance: set[str] = field(default_factory=set)

    # Lower rank is better. None means that the blocker did not
    # provide a rank.
    ranks: dict[str, int] = field(default_factory=dict)

    def add_provenance(
        self,
        blocker: str,
        rank: int | None = None,
    ) -> None:
        """Add a blocker that discovered this candidate."""
        self.blocker_provenance.add(blocker)

        if rank is not None:
            self.ranks[blocker] = rank

    @property
    def num_blockers(self) -> int:
        """Number of distinct blockers that found this candidate."""
        return len(self.blocker_provenance)

    @property
    def best_rank(self) -> int | None:
        """Best rank across all blockers that supplied a rank."""
        if not self.ranks:
            return None

        return min(self.ranks.values())

    def to_dict(self) -> dict:
        """Convert the candidate into a serializable dictionary."""
        return {
            "source1_entity_id": self.source1_entity_id,
            "candidate_entity_id": self.candidate_entity_id,
            "source": self.source,
            "blocker_provenance": sorted(self.blocker_provenance),
            "ranks": dict(self.ranks),
            "num_blockers": self.num_blockers,
            "best_rank": self.best_rank,
        }