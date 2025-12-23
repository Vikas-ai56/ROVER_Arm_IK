"""Utility functions for Lie groups."""

import numpy as np


def get_epsilon(dtype: np.dtype) -> float:
    """Get numerical epsilon based on dtype.
    
    Args:
        dtype: NumPy data type.
        
    Returns:
        Appropriate epsilon value.
    """
    return {
        np.dtype("float32"): 1e-5,
        np.dtype("float64"): 1e-10,
    }.get(dtype, 1e-10)


def skew(x: np.ndarray) -> np.ndarray:
    """Compute skew-symmetric matrix from 3D vector.
    
    Args:
        x: 3D vector.
        
    Returns:
        3x3 skew-symmetric matrix.
    """
    assert x.shape == (3,), f"Expected 3D vector, got shape {x.shape}"
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=x.dtype,
    )
