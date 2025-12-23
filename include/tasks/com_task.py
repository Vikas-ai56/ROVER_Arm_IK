"""Center-of-mass task implementation for Pinocchio."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pinocchio as pin

if TYPE_CHECKING:
    from ..configuration import Configuration

from ..exceptions import TargetNotSet, TaskDefinitionError
from .task import Task


class ComTask(Task):
    """Regulate the center-of-mass (CoM) of a robot.

    This task controls the 3D position of the robot's center of mass.
    Useful for balance control and whole-body motion planning.

    Attributes:
        target_com: Target position of the CoM in world frame (3D vector).

    Example:
        >>> com_task = ComTask(cost=1.0)
        >>> com_desired = np.array([0.0, 0.0, 0.5])
        >>> com_task.set_target(com_desired)
    """

    k: int = 3
    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize CoM task.
        
        Args:
            cost: Cost for CoM error (scalar or 3D vector).
            gain: Task gain in [0, 1].
            lm_damping: Levenberg-Marquardt damping.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.target_com = None
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost of the CoM task.

        Args:
            cost: Scalar or 3D vector of costs.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"or ({self.k},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_com: npt.ArrayLike) -> None:
        """Set the target CoM position in the world frame.

        Args:
            target_com: 3D vector representing desired CoM position.
        """
        target_com = np.atleast_1d(target_com)
        if target_com.ndim != 1 or target_com.shape[0] != self.k:
            raise TaskDefinitionError(
                f"Expected target CoM to have shape ({self.k},) but got "
                f"{target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: "Configuration") -> None:
        """Set the target CoM from a given robot configuration.

        Args:
            configuration: Robot configuration q.
        """
        # Compute center of mass for current configuration
        com = pin.centerOfMass(configuration.model, configuration.data)
        self.set_target(com)

    def compute_error(self, configuration: "Configuration") -> np.ndarray:
        """Compute the CoM task error.

        The task error is the difference between current CoM and target CoM:
        e(q) = com(q) - com_target

        Args:
            configuration: Robot configuration q.

        Returns:
            Center-of-mass task error vector e(q).
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        
        # Get current center of mass
        current_com = pin.centerOfMass(configuration.model, configuration.data)
        
        return current_com - self.target_com

    def compute_jacobian(self, configuration: "Configuration") -> np.ndarray:
        """Compute the Jacobian of the CoM task error.

        The Jacobian is the derivative of the CoM position with respect to
        the generalized coordinates:
        J(q) = ∂com(q)/∂q

        Args:
            configuration: Robot configuration q.

        Returns:
            Task jacobian J(q) of shape (3, nv).
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        
        # Compute CoM Jacobian using Pinocchio
        # Returns Jacobian of shape (3, nv)
        J_com = pin.jacobianCenterOfMass(configuration.model, configuration.data)
        
        return J_com
