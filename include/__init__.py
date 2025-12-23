"""Pinocchio-based inverse kinematics solver.

A stateless, simulator-independent IK solver inspired by Mink's architecture.
Uses Pinocchio for kinematics and OSQP for quadratic programming.
"""

from .configuration import Configuration
from .constants import (
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_GAIN,
    DEFAULT_QP_SOLVER,
    DEFAULT_TOLERANCE,
    EPSILON_FLOAT32,
    EPSILON_FLOAT64,
    QP_EPS_ABS,
    QP_EPS_REL,
    SUPPORTED_FRAMES,
)
from .exceptions import (
    IKError,
    InvalidDamping,
    InvalidFrame,
    InvalidGain,
    InvalidTarget,
    LimitDefinitionError,
    NoSolutionFound,
    NotWithinConfigurationLimits,
    TargetNotSet,
    TaskDefinitionError,
)
from .lie import SE3, SO3, MatrixLieGroup
from .limits import ConfigurationLimit, Constraint, Limit, VelocityLimit
from .solve_ik import build_ik, solve_ik
from .tasks import (
    BaseTask,
    DampingTask,
    FrameTask,
    Objective,
    PostureTask,
    Task,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Configuration",
    # Solver
    "build_ik",
    "solve_ik",
    # Tasks
    "BaseTask",
    "DampingTask",
    "FrameTask",
    "Objective",
    "PostureTask",
    "Task",
    # Limits
    "ConfigurationLimit",
    "Constraint",
    "Limit",
    "VelocityLimit",
    # Lie groups
    "MatrixLieGroup",
    "SE3",
    "SO3",
    # Exceptions
    "IKError",
    "InvalidDamping",
    "InvalidFrame",
    "InvalidGain",
    "InvalidTarget",
    "LimitDefinitionError",
    "NoSolutionFound",
    "NotWithinConfigurationLimits",
    "TargetNotSet",
    "TaskDefinitionError",
    # Constants
    "DEFAULT_DAMPING",
    "DEFAULT_DT",
    "DEFAULT_GAIN",
    "DEFAULT_QP_SOLVER",
    "DEFAULT_TOLERANCE",
    "EPSILON_FLOAT32",
    "EPSILON_FLOAT64",
    "QP_EPS_ABS",
    "QP_EPS_REL",
    "SUPPORTED_FRAMES",
]
