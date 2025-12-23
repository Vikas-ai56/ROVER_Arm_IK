"""Posture task implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class PostureTask(Task):
    """Regulate joint angles towards a target posture.

    Often used with a low priority in the task stack, this task acts like a regularizer,
    biasing the solution toward a specific joint configuration.

    Attributes:
        target_q: Target configuration q*, of shape (nq,). Units are
            radians for revolute joints and meters for prismatic joints.

    Example:
        >>> model = pin.buildModelFromUrdf("robot.urdf")
        >>> posture_task = PostureTask(model, cost=1e-3)
        >>> q_desired = pin.neutral(model)
        >>> posture_task.set_target(q_desired)
    """

    target_q: Optional[np.ndarray]

    def __init__(
        self,
        model: pin.Model,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize posture task.
        
        Args:
            model: Pinocchio model.
            cost: Cost for posture error (scalar or nv-dimensional vector).
            gain: Task gain in [0, 1].
            lm_damping: Levenberg-Marquardt damping.
        """
        super().__init__(
            cost=np.zeros((model.nv,)),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_q = None
        self.nq = model.nq
        self.nv = model.nv
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set posture cost.
        
        Args:
            cost: Scalar or nv-dimensional vector of costs.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.nv):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(identical cost for all dofs) or ({self.nv},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost[:self.nv] = cost

    def set_target(self, target_q: npt.ArrayLike) -> None:
        """Set the target posture.

        Args:
            target_q: A vector of shape (nq,) representing the desired joint
                configuration.
        """
        target_q = np.atleast_1d(target_q)
        if target_q.ndim != 1 or target_q.shape[0] != self.nq:
            raise InvalidTarget(
                f"Expected target posture to have shape ({self.nq},) but got "
                f"{target_q.shape}"
            )
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target posture by extracting it from the current configuration.

        Args:
            configuration: Robot configuration q.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the posture task error.

        Args:
            configuration: Robot configuration q.

        Returns:
            Posture error e(q) = target_q - q of shape (nv,).
        """
        if self.target_q is None:
            raise TargetNotSet(self.__class__.__name__)

        # Use Pinocchio's difference function to handle special cases
        error = pin.difference(configuration.model, configuration.q, self.target_q)
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the posture task Jacobian.

        For posture task, the Jacobian is the identity matrix since
        joint velocities directly affect joint positions.

        Args:
            configuration: Robot configuration q.

        Returns:
            Identity matrix of shape (nv, nv).
        """
        return np.eye(self.nv)
