
import pytest

from business_entity_resol.blocking.candidate_union import CandidateUnion
from business_entity_resol.evaluation.blocking_evaluation import (
    candidate_union_to_mapping,
    evaluate_candidate_union,
)


def test_candidate_union_to_mapping():
    union = CandidateUnion()

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S2-001",
        source="S2",
        blocker="exact",
    )

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S3-001",
        source="S3",
        blocker="tfidf",
        rank=1,
    )

    result = candidate_union_to_mapping(union)

    assert result == {
        "S1-001": {
            "S2-001",
            "S3-001",
        }
    }


def test_evaluate_candidate_union():
    union = CandidateUnion()

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S2-001",
        source="S2",
        blocker="exact",
    )

    union.add(
        source1_entity_id="S1-001",
        candidate_entity_id="S3-001",
        source="S3",
        blocker="tfidf",
        rank=1,
    )

    union.add(
        source1_entity_id="S1-002",
        candidate_entity_id="S2-002",
        source="S2",
        blocker="numeric",
        rank=1,
    )

    ground_truth = {
        "S1-001": {
            "S2-001",
            "S3-001",
        },
        "S1-002": {
            "S2-002",
        },
    }

    result = evaluate_candidate_union(
        ground_truth,
        union,
    )

    assert result["link_recall"] == 1.0
    assert result["complete_entity_recall"] == 1.0

    per_entity = result["per_entity_recall"]

    assert per_entity["S1-001"] == pytest.approx(1.0)
    assert per_entity["S1-002"] == pytest.approx(1.0)

    stats = result["candidate_count_stats"]

    assert stats["mean"] == pytest.approx(1.5)
    assert stats["median"] == pytest.approx(1.5)
    assert stats["max"] == 2
