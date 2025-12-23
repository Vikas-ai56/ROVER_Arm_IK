"""Damping task implementation."""

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
    """L2-regularization on joint velocities (a.k.a. velocity damping).

    This task, typically used with a low priority in the task stack, adds a
    Levenberg-Marquardt term to the quadratic program, favoring minimum-norm
    joint velocities in redundant or near-singular situations. Formally, it
    contributes the following term to the quadratic program:
        (1/2) * dq^T * Lambda * dq

    where dq is the vector of joint displacements and Lambda is a diagonal 
    matrix of per-DoF damping weights specified via cost. A larger cost
    reduces motion in that DoF.

    With no other active tasks, the robot remains at rest.

    Example:
        >>> model = pin.buildModelFromUrdf("robot.urdf")
        >>> # Uniform damping across all degrees of freedom
        >>> damping_task = DampingTask(model, cost=1.0)
        >>> # Custom damping
        >>> cost = np.zeros(model.nv)
        >>> cost[:3] = 1.0  # High damping for first 3 joints
        >>> cost[3:] = 0.1  # Lower damping for remaining joints
        >>> damping_task = DampingTask(model, cost)
    """

    def __init__(self, model: pin.Model, cost: npt.ArrayLike):
        """Initialize damping task.
        
        Args:
            model: Pinocchio model.
            cost: Damping cost (scalar or nv-dimensional vector).
        """
        super().__init__(model, cost, gain=0.0, lm_damping=0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the damping task error.

        The damping task does not chase a reference; its desired joint velocity
        is identically zero, so the task error is always zero.

        Args:
            configuration: Robot configuration q.

        Returns:
            Zero vector of length nv.
        """
        return np.zeros(configuration.nv)
