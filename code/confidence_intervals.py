"""
confidence_intervals.py

Provides functions to compute confidence intervals using either the
Student’s t-distribution (for small samples) or the normal distribution
(for larger samples), with an initializer that dispatches appropriately.
"""

import numpy as np
import scipy.stats as st

def confidence_interval_t(data, confidence=0.95):
    """
    Calculate the t-distribution based confidence interval for a given dataset and confidence level.
    """
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    t = st.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    confidence_interval = np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, confidence_interval, margin_of_error

def confidence_interval_normal(data, confidence=0.95):
    """
    Calculate the normal distribution based confidence interval for a given dataset and confidence level.
    """
    data_array = 1.0 * np.array(data)
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    z = norm().ppf((1 + confidence) / 2.)
    margin_of_error = sample_standard_error * z
    confidence_interval = np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, confidence_interval, margin_of_error

def confidence_interval_init(data, confidence=0.95):
    """
    Initialize confidence interval calculations for a dataset, handling multidimensional data and selecting
    the appropriate method based on sample size.
    """
    data_array = 1.0 * np.array(data)
    dimensions = data_array.shape
    if len(dimensions) > 1:
        rows, columns = dimensions[0], dimensions[1]
        if columns <= 30:
            method = confidence_interval_t
        else:
            method = confidence_interval_normal
        sample_mean_array, confidence_interval_array, margin_of_error_array = method(data_array[0], confidence)
        for row in range(1, rows):
            sample_mean_new_row, confidence_interval_new_row, margin_of_error_new_row = method(data_array[row], confidence)
            sample_mean_array = np.append(sample_mean_array, sample_mean_new_row)
            confidence_interval_array = np.vstack((confidence_interval_array, confidence_interval_new_row))
            margin_of_error_array = np.append(margin_of_error_array, margin_of_error_new_row)
        return sample_mean_array, confidence_interval_array, margin_of_error_array
    else:
        if len(data_array) <= 30:
            return confidence_interval_t(data_array, confidence)
        else:
            return confidence_interval_normal(data_array, confidence)