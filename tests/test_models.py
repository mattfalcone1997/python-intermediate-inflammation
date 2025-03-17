"""Tests for statistics functions within the Model layer."""

import numpy.testing as npt

from inflammation.models import (daily_mean,
                                 daily_max,
                                 daily_min)
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
