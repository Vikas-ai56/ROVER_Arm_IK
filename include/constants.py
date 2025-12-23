"""Constants used throughout the IK solver."""

import numpy as np

# Supported reference frames in Pinocchio
SUPPORTED_FRAMES = ("frame",)  # Pinocchio uses generic "frame" type

# Numerical tolerances
DEFAULT_TOLERANCE = 1e-6
EPSILON_FLOAT32 = 1e-5
EPSILON_FLOAT64 = 1e-10

# Default solver parameters
DEFAULT_DAMPING = 1e-12
DEFAULT_GAIN = 1.0
DEFAULT_DT = 0.001

# QP solver settings
DEFAULT_QP_SOLVER = "osqp"
QP_EPS_ABS = 1e-4
QP_EPS_REL = 1e-4


def get_epsilon(dtype: np.dtype) -> float:
    """Get numerical epsilon for a given dtype."""
    return {
        np.dtype("float32"): EPSILON_FLOAT32,
        np.dtype("float64"): EPSILON_FLOAT64,
    }.get(dtype, EPSILON_FLOAT64)
