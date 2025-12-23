"""Configuration space of a robot model.

The Configuration class encapsulates a Pinocchio model and data,
offering easy access to frame transforms and frame Jacobians.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3


class Configuration:
    """Encapsulates a Pinocchio model and data for convenient access to kinematic quantities.

    This class provides methods to access and update the kinematic quantities of a robot
    model, such as frame transforms and Jacobians. It performs forward kinematics at every
    time step, ensuring up-to-date information about the robot's state.

    Key functionalities include:
    * Running forward kinematics to update the state.
    * Checking configuration limits.
    * Computing Jacobians for different frames.
    * Retrieving frame transforms relative to the world frame.
    * Integrating velocities to update configurations.
    """

    def __init__(
        self,
        model: pin.Model,
        q: Optional[np.ndarray] = None,
    ):
        """Constructor.

        Args:
            model: Pinocchio model.
            q: Configuration to initialize from. If None, the configuration is
                initialized to the neutral configuration.
        """
        self.model = model
        self.data = model.createData()
        
        if q is None:
            q = pin.neutral(model)
        
        self.update(q=q)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        q: Optional[np.ndarray] = None,
    ) -> "Configuration":
        """Create a Configuration from a URDF file.

        Args:
            urdf_path: Path to URDF file.
            q: Optional initial configuration.

        Returns:
            Configuration instance.
        """
        model = pin.buildModelFromUrdf(urdf_path)
        return cls(model, q)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Run forward kinematics.

        Args:
            q: Optional configuration vector to override internal data.q with.
        """
        if q is not None:
            self.data.q = q.copy()
        
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, self.data.q)
        pin.updateFramePlacements(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        """Get current configuration."""
        return self.data.q

    @property
    def nq(self) -> int:
        """Configuration space dimension."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Tangent space dimension."""
        return self.model.nv

    def check_limits(self, tol: float = consts.DEFAULT_TOLERANCE, safety_break: bool = True) -> None:
        """Check that the current configuration is within bounds.

        Args:
            tol: Tolerance in [rad] or [m].
            safety_break: If True, stop execution and raise an exception if the current
                configuration is outside limits. If False, print a warning and continue.

        Raises:
            NotWithinConfigurationLimits: If the current configuration is outside
                the joint limits.
        """
        q = self.data.q
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit

        for i in range(self.model.nq):
            if q[i] < q_min[i] - tol or q[i] > q_max[i] + tol:
                if safety_break:
                    raise exceptions.NotWithinConfigurationLimits(
                        joint_id=i,
                        value=q[i],
                        lower=q_min[i],
                        upper=q_max[i],
                        model=self.model,
                    )
                else:
                    logging.warning(
                        f"Value {q[i]:.4f} at index {i} is outside of its limits: "
                        f"[{q_min[i]:.4f}, {q_max[i]:.4f}]"
                    )

    def get_frame_jacobian(self, frame_name: str) -> np.ndarray:
        """Compute the Jacobian matrix of a frame velocity.

        The Jacobian relates the frame velocity to joint velocities:
        v_frame = J * dq

        Args:
            frame_name: Name of the frame in the URDF.

        Returns:
            Jacobian of the frame (6, nv) - [linear velocity; angular velocity]
        """
        if not self.model.existFrame(frame_name):
            raise exceptions.InvalidFrame(frame_name, self.model)

        frame_id = self.model.getFrameId(frame_name)
        
        # Compute the Jacobian in the local frame
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.data.q,
            frame_id,
            pin.ReferenceFrame.LOCAL
        )
        
        return J

    def get_transform_frame_to_world(self, frame_name: str) -> SE3:
        """Get the pose of a frame at the current configuration.

        Args:
            frame_name: Name of the frame in the URDF.

        Returns:
            The pose of the frame in the world frame.
        """
        if not self.model.existFrame(frame_name):
            raise exceptions.InvalidFrame(frame_name, self.model)

        frame_id = self.model.getFrameId(frame_name)
        
        # Get the frame placement (SE3 transform)
        placement = self.data.oMf[frame_id]
        
        return SE3.from_pinocchio_se3(placement)

    def get_transform(
        self,
        source_name: str,
        dest_name: str,
    ) -> SE3:
        """Get relative transform between two frames.

        Args:
            source_name: Name of source frame.
            dest_name: Name of destination frame.

        Returns:
            Transform from source to destination.
        """
        T_world_source = self.get_transform_frame_to_world(source_name)
        T_world_dest = self.get_transform_frame_to_world(dest_name)
        
        return T_world_source.inverse().multiply(T_world_dest)

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        """Integrate a tangent velocity and update configuration in place.

        Args:
            velocity: Velocity in tangent space (nv,).
            dt: Integration timestep in [s].
        """
        self.data.q = pin.integrate(self.model, self.data.q, velocity * dt)
        self.update()

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a tangent velocity and return new configuration.

        Args:
            velocity: Velocity in tangent space (nv,).
            dt: Integration timestep in [s].

        Returns:
            New configuration.
        """
        return pin.integrate(self.model, self.data.q, velocity * dt)
