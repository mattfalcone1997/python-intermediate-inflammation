"""Tests for statistics functions within the Model layer."""

import numpy.testing as npt
import numpy as np
from inflammation.models import (daily_mean,
                                 daily_max,
                                 daily_min,
                                 patient_normalise)
import pytest

@pytest.mark.parametrize("test, expected",
                         [
                             ([[0, 0],[0, 0],[0, 0]], [0,0]),
                             ([[1, 2],[3, 4],[5, 6]], [3,4])
                         ])
def test_daily_mean_zeros(test, expected):
    """Test that mean function works for a range of inputs."""
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [4, 6, 9]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test that daily_max function works for a range of inputs."""
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test), expected)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [4, 2, 5], [1, 6, 2], [4, 1, 9] ], [1, 1, 2]),
        ([ [4, -2, 5], [1, -6, 2], [-4, -1, 9] ], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test that daily_min function works for a range of inputs."""
    npt.assert_array_equal(daily_min(test), expected)

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1/3, 2/3, 1], [2/3, 5/6, 1], [7/9, 8/9, 1]],
            None,
        ),
        (
            [[0, 0, 0], [4, 5, 6], [7, 8, 9]],
            [[0, 0, 0], [2/3, 5/6, 1], [7/9, 8/9, 1]],
            None,
        ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test that patient_normalise function works for a range of inputs."""
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(test)
    else:
        npt.assert_allclose(patient_normalise(test), expected, rtol=1e-2, atol=1e-12)
