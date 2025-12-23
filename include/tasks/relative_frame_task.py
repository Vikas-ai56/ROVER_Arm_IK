"""Relative frame task implementation for Pinocchio."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..configuration import Configuration

from ..exceptions import TargetNotSet, TaskDefinitionError
from ..lie import SE3
from .task import Task


class RelativeFrameTask(Task):
    """Regulate the pose of a frame relative to another frame.

    This task controls the relative pose between two frames on the robot.
    Useful for maintaining relative configurations (e.g., keeping end-effector
    at a fixed pose relative to the robot base or another link).

    Attributes:
        frame_name: Name of the frame to regulate.
        root_name: Name of the root frame (reference frame).
        transform_target_to_root: Target pose of frame relative to root.

    Example:
        >>> task = RelativeFrameTask(
        ...     frame_name="end_effector",
        ...     root_name="base_link",
        ...     position_cost=1.0,
        ...     orientation_cost=1.0,
        ... )
        >>> # Set target relative pose
        >>> target_se3 = SE3.from_translation(np.array([0.1, 0.0, 0.2]))
        >>> task.set_target(target_se3)
    """

    k: int = 6
    transform_target_to_root: Optional[SE3]

    def __init__(
        self,
        frame_name: str,
        root_name: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Initialize relative frame task.
        
        Args:
            frame_name: Name of the frame to control.
            root_name: Name of the reference frame.
            position_cost: Cost for position error (scalar or 3D vector).
            orientation_cost: Cost for orientation error (scalar or 3D vector).
            gain: Task gain in [0, 1].
            lm_damping: Levenberg-Marquardt damping.
        """
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.root_name = root_name
        self.transform_target_to_root = None

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

    def set_target(self, transform_target_to_root: SE3) -> None:
        """Set the target relative pose.

        Args:
            transform_target_to_root: Transform from task frame to root frame.
        """
        self.transform_target_to_root = transform_target_to_root.copy()

    def set_target_from_configuration(self, configuration: "Configuration") -> None:
        """Set the target from current configuration.

        Args:
            configuration: Robot configuration q.
        """
        # Get current relative transform
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name
        )
        transform_root_to_world = configuration.get_transform_frame_to_world(
            self.root_name
        )
        
        # Compute relative transform: T_frame_to_root = T_root_to_world^-1 * T_frame_to_world
        transform_frame_to_root = transform_root_to_world.inverse() @ transform_frame_to_world
        
        self.set_target(transform_frame_to_root)

    def compute_error(self, configuration: "Configuration") -> np.ndarray:
        """Compute the relative frame task error.

        The error is the SE3 difference between current relative pose
        and target relative pose, expressed as a 6D twist.

        Args:
            configuration: Robot configuration q.

        Returns:
            Task error vector e(q) of shape (6,).
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        # Get current relative transform
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name
        )
        transform_root_to_world = configuration.get_transform_frame_to_world(
            self.root_name
        )
        
        transform_frame_to_root = transform_root_to_world.inverse() @ transform_frame_to_world

        # Compute SE3 error
        return transform_frame_to_root.rminus(self.transform_target_to_root)

    def compute_jacobian(self, configuration: "Configuration") -> np.ndarray:
        """Compute the Jacobian of the relative frame task.

        The Jacobian maps joint velocities to the relative twist between
        the two frames.

        Args:
            configuration: Robot configuration q.

        Returns:
            Task jacobian J(q) of shape (6, nv).
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        # Get Jacobians for both frames in their local frames
        J_frame = configuration.get_frame_jacobian(self.frame_name)
        J_root = configuration.get_frame_jacobian(self.root_name)

        # Get transforms
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame_name
        )
        transform_root_to_world = configuration.get_transform_frame_to_world(
            self.root_name
        )
        
        transform_frame_to_root = transform_root_to_world.inverse() @ transform_frame_to_world
        transform_frame_to_target = (
            self.transform_target_to_root.inverse() @ transform_frame_to_root
        )

        # Relative Jacobian: J_rel = J_log(T_error) * (J_frame - Ad(T_frame_to_root^-1) * J_root)
        return transform_frame_to_target.jlog() @ (
            J_frame - transform_frame_to_root.inverse().adjoint() @ J_root
        )
