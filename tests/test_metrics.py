import pytest

from business_entity_resol.evaluation.entity_f05 import (
    macro_f05,
    per_entity_f05,
    score_entity_f05,
)


def test_perfect_match():
    result = score_entity_f05(
        {"S2-001", "S3-001"},
        {"S2-001", "S3-001"},
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f05 == 1.0
    assert result.true_positive == 2
    assert result.false_positive == 0
    assert result.false_negative == 0


def test_partial_match():
    result = score_entity_f05(
        {"S2-001", "S3-001"},
        {"S2-001", "S2-999"},
    )

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f05 == pytest.approx(0.5)


def test_missed_match():
    result = score_entity_f05(
        {"S2-001"},
        set(),
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f05 == 0.0
    assert result.false_negative == 1


def test_correct_singleton():
    result = score_entity_f05(
        set(),
        set(),
    )

    assert result.f05 == 1.0


def test_false_match_for_singleton():
    result = score_entity_f05(
        set(),
        {"S2-001"},
    )

    assert result.f05 == 0.0
    assert result.false_positive == 1


def test_duplicate_prediction_ids_do_not_change_set_metric():
    result = score_entity_f05(
        {"S2-001"},
        ["S2-001", "S2-001"],
    )

    assert result.f05 == 1.0


def test_macro_f05():
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    predictions = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    assert macro_f05(ground_truth, predictions) == 1.0


def test_macro_f05_includes_singletons():
    ground_truth = {
        "S1-001": {"S2-001"},
        "S1-002": set(),
    }

    predictions = {
        "S1-001": {"S2-001"},
        "S1-002": {"S2-999"},
    }

    assert macro_f05(
        ground_truth,
        predictions,
    ) == pytest.approx(0.5)


def test_missing_prediction_entity_raises():
    with pytest.raises(ValueError):
        macro_f05(
            {"S1-001": {"S2-001"}},
            {},
        )


def test_extra_prediction_entity_raises():
    with pytest.raises(ValueError):
        macro_f05(
            {"S1-001": {"S2-001"}},
            {
                "S1-001": {"S2-001"},
                "S1-999": set(),
            },
        )


def test_per_entity_scores():
    results = per_entity_f05(
        {
            "S1-001": {"S2-001"},
            "S1-002": set(),
        },
        {
            "S1-001": {"S2-001"},
            "S1-002": set(),
        },
    )

    assert results["S1-001"].f05 == 1.0
    assert results["S1-002"].f05 == 1.0