"""
conftest.py tier-1 fixtures shared across the entire test suite.

All fixtures here use synthetic data so the suite runs on any machine
without requiring a downloaded or processed ABIDE dataset.

To run the full suite:
    pytest -v
"""

import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data: test needs pre-computed ABIDE files in product/data/ "
        "(run the pipeline scripts first)",
    )


@pytest.fixture
def small_synthetic_dataset():
    """Small synthetic dataset for fast testing."""
    np.random.seed(42)
    X = np.random.randn(100, 19900)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def separable_synthetic_dataset():
    """Linearly separable dataset for testing learning."""
    np.random.seed(42)
    X_class0 = np.random.randn(50, 19900) - 1.0
    X_class1 = np.random.randn(50, 19900) + 1.0
    X = np.vstack([X_class0, X_class1])
    y = np.array([0]*50 + [1]*50)
    return X, y
