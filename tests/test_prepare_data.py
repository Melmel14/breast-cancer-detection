"""Tests for the data preparation helpers.

prepare_data.py is the highest-risk script in the project, because a fault here
does not crash. It quietly attaches the wrong pathology label to an image, the
network trains on noise, and the results look plausible but mean nothing. These
tests exercise the two helpers that decide where a label comes from.
"""

import os

import pandas as pd

from prepare_data import find_dir, pick_column


def test_find_dir_locates_a_folder_at_the_root(tmp_path):
    os.makedirs(tmp_path / "csv")
    assert find_dir(str(tmp_path), "csv") == str(tmp_path / "csv")


def test_find_dir_locates_a_folder_one_level_down(tmp_path):
    """The Kaggle download sometimes nests csv/ and jpeg/ inside a wrapper
    folder, so the lookup has to survive both layouts."""
    os.makedirs(tmp_path / "CBIS-DDSM" / "jpeg")
    found = find_dir(str(tmp_path), "jpeg")
    assert found is not None
    assert found.endswith(os.path.join("CBIS-DDSM", "jpeg"))


def test_find_dir_returns_none_when_absent(tmp_path):
    """Returning None rather than raising is what lets the script print a
    readable diagnostic instead of a traceback."""
    assert find_dir(str(tmp_path), "csv") is None


def test_pick_column_matches_ignoring_case_and_spacing():
    """Column names have shifted between releases of this dataset, so the
    matching is deliberately tolerant."""
    df = pd.DataFrame(columns=[" Pathology ", "image file path"])
    assert pick_column(df, "pathology") == " Pathology "


def test_pick_column_returns_none_for_an_unknown_column():
    df = pd.DataFrame(columns=["pathology"])
    assert pick_column(df, "does_not_exist") is None
