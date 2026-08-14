"""Tests for the clinical metric computation.

These run without TensorFlow, without images and without a trained model, so the
arithmetic that decides how the system reports a missed cancer can be checked in
under a second on every change.
"""

import pytest

from metrics import compute_metrics


def test_perfect_classifier_scores_one():
    """No errors in either direction means every rate is 1.0."""
    result = compute_metrics([[50, 0], [0, 50]])
    assert result["accuracy"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0


def test_sensitivity_penalises_missed_malignant_cases():
    """A model that calls every case benign has perfect specificity and zero
    sensitivity. This is the failure mode accuracy alone would hide."""
    result = compute_metrics([[80, 0], [20, 0]])
    assert result["specificity"] == 1.0
    assert result["sensitivity"] == 0.0
    assert result["accuracy"] == pytest.approx(0.8)


def test_matches_the_recorded_baseline_run():
    """Regression test against the real CBIS-DDSM baseline run.

    These are the counts from the saved confusion matrix, so if a later change
    to the pipeline silently alters how the matrix is read, this test catches it.
    """
    result = compute_metrics([[87, 73], [48, 121]])
    assert result["accuracy"] == pytest.approx(0.632, abs=0.001)
    assert result["sensitivity"] == pytest.approx(0.716, abs=0.001)
    assert result["specificity"] == pytest.approx(0.544, abs=0.001)


def test_counts_are_unpacked_in_the_right_order():
    """Guards against transposing the matrix, which would swap sensitivity and
    specificity and make the system look safer than it is."""
    result = compute_metrics([[1, 2], [3, 4]])
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 2, 3, 4)


def test_empty_class_does_not_raise():
    """A degenerate split with no malignant cases returns 0.0 rather than
    dividing by zero and killing the run at the final step."""
    result = compute_metrics([[10, 0], [0, 0]])
    assert result["sensitivity"] == 0.0
    assert result["specificity"] == 1.0
