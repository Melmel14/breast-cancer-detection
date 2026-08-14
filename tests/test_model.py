"""Tests for the network and the inference contract.

These import TensorFlow and so are slower than the rest of the suite. They do not
train anything: they check the shape of the contract between the model and the
code around it, which is where a refactor is most likely to break something
without any visible error.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from prototype import build_model, make_synthetic


def test_model_outputs_one_probability_per_image():
    """A single sigmoid output per image. If this shape changes, every
    downstream metric silently misreads the predictions."""
    model = build_model(64)
    batch = np.zeros((4, 64, 64, 1), dtype="float32")
    output = model.predict(batch, verbose=0)
    assert output.shape == (4, 1)


def test_model_output_is_a_valid_probability():
    """The sigmoid must bound the output to [0, 1], since predict.py reports it
    to the user as a confidence."""
    model = build_model(64)
    batch = np.random.rand(4, 64, 64, 1).astype("float32")
    output = model.predict(batch, verbose=0)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_synthetic_generator_produces_both_classes():
    """The synthetic fallback has to contain real, learnable structure,
    otherwise a passing pipeline test would prove nothing."""
    images, labels = make_synthetic(40, 64)
    assert len(images) == len(labels) == 40
    assert set(np.unique(labels)) == {0, 1}
