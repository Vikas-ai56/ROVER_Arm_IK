"""Frame task implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..lie import SE3
from .task import Task


class FrameTask(Task):
    """Regulate the position and orientation of a frame of interest on the robot.

    Attributes:
        frame_name: Name of the frame to regulate.
        transform_target_to_world: Target pose of the frame in the world frame.

    Example:
        >>> frame_task = FrameTask(
        ...     frame_name="end_effector",
        ...     position_cost=1.0,
        ...     orientation_cost=1.0,
        ... )
        >>> transform_target_to_world = SE3.from_translation(np.array([0.5, 0.2, 0.3]))
        >>> frame_task.set_target(transform_target_to_world)
    """

    k: int = 6
    transform_target_to_world: Optional[SE3]

    def __init__(
        self,
        frame_name: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize frame task.
        
        Args:
            frame_name: Name of the frame in the URDF.
            position_cost: Cost for position error (scalar or 3D vector).
            orientation_cost: Cost for orientation error (scalar or 3D vector).
            gain: Task gain in [0, 1].
            lm_damping: Levenberg-Marquardt damping.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.transform_target_to_world = None

        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        """Set position cost.
        
        Args:
            position_cost: Scalar or 3D vector of position costs.
        """
        position_cost = np.atleast_1d(position_cost)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                "1 (identical cost for all coordinates) or (3,) but got "
                f"{position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        """Set orientation cost.
        
        Args:
            orientation_cost: Scalar or 3D vector of orientation costs.
        """
        orientation_cost = np.atleast_1d(orientation_cost)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                "shape 1 (identical cost for all coordinates) or (3,) but got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be >= 0"
            )
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_world: SE3) -> None:
        """Set the target pose.

        Args:
            transform_target_to_world: Transform from the task target frame to the
                world frame.
        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration q.
        """
        self.set_target(configuration.get_transform_frame_to_world(self.frame_name))

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the frame task error.

        This error is a twist e(q) in se(3) expressed in the local frame.
        The error points FROM current TO target, so minimizing ||J*dq + gain*e||
        drives the frame toward the target.

        Args:
            configuration: Robot configuration q.

        Returns:
            Frame task error vector e(q) of shape (6,).
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(self.__class__.__name__)

        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name
        )
        
        # Compute error as log(T_current^{-1} * T_target)
        # This gives the twist FROM current frame TO target
        return transform_frame_to_world.minus(self.transform_target_to_world)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the frame task Jacobian.

        Args:
            configuration: Robot configuration q.

        Returns:
            Jacobian matrix J(q) of shape (6, nv).
        """
        return configuration.get_frame_jacobian(self.frame_name)
