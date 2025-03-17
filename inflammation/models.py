"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.

Functions
    load_csv - returns numpy array with csv contents
    daily_mean - returns the daily mean of the inflammation data
    daily_max - returns the daily max of the inflammation data
    daily_min - returns the daily min of the inflammation data
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV.

    :param filename: Filename of CSV to load
    :returns: Numpy array of csv data
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: 2D array with inflammation data (each row is a single patient across all days).
    :returns: array of mean daily inflammation across all patients.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: 2D array with inflammation data (each row is a single patient across all days).
    :returns: array of daily maximum inflammation across all patients.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: 2D array with inflammation data (each row is a single patient across all days).
    :returns: array of daily minimum inflammation across all patients.
    """
    return np.min(data, axis=0)
