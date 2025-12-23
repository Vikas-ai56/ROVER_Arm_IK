"""SO(3): Special Orthogonal group for 3D rotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .base import MatrixLieGroup
from .utils import get_epsilon, skew

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # [x, y, z, w]


@dataclass(frozen=True)
class SO3(MatrixLieGroup):
    """Special orthogonal group for 3D rotations.

    Internal parameterization is quaternion [x, y, z, w].
    Tangent parameterization is (omega_x, omega_y, omega_z).
    """

    quat: np.ndarray  # [x, y, z, w]
    matrix_dim: int = 3
    parameters_dim: int = 4
    tangent_dim: int = 3
    space_dim: int = 3

    def __repr__(self) -> str:
        quat = np.round(self.quat, 5)
        return f"{self.__class__.__name__}(quat={quat})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SO3):
            return NotImplemented
        return np.allclose(self.quat, other.quat)

    def parameters(self) -> np.ndarray:
        return self.quat

    def copy(self) -> SO3:
        return SO3(quat=self.quat.copy())

    @classmethod
    def identity(cls) -> SO3:
        return SO3(quat=_IDENTITY_QUAT.copy())

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SO3:
        """Create SO3 from rotation matrix.
        
        Args:
            matrix: 3x3 rotation matrix.
            
        Returns:
            SO3 instance.
        """
        assert matrix.shape == (3, 3)
        quat = pin.Quaternion(matrix).coeffs()  # Returns [x, y, z, w]
        return SO3(quat=quat)

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float) -> SO3:
        """Create SO3 from roll-pitch-yaw angles.
        
        Args:
            roll: Roll angle in radians.
            pitch: Pitch angle in radians.
            yaw: Yaw angle in radians.
            
        Returns:
            SO3 instance.
        """
        quat = pin.Quaternion(pin.rpy.rpyToMatrix(roll, pitch, yaw)).coeffs()
        return SO3(quat=quat)

    def as_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        return pin.Quaternion(self.quat[3], self.quat[0], self.quat[1], self.quat[2]).toRotationMatrix()

    @classmethod
    def exp(cls, tangent: np.ndarray) -> SO3:
        """Exponential map from tangent space to SO(3).
        
        Args:
            tangent: 3D angular velocity vector.
            
        Returns:
            SO3 instance.
        """
        assert tangent.shape == (3,)
        theta = np.linalg.norm(tangent)
        
        if theta < get_epsilon(tangent.dtype):
            # Small angle approximation
            return SO3.identity()
        
        axis = tangent / theta
        quat = np.zeros(4)
        quat[:3] = np.sin(theta / 2) * axis
        quat[3] = np.cos(theta / 2)
        
        return SO3(quat=quat)

    def log(self) -> np.ndarray:
        """Logarithm map from SO(3) to tangent space.
        
        Returns:
            3D angular velocity vector.
        """
        quat_pin = pin.Quaternion(self.quat[3], self.quat[0], self.quat[1], self.quat[2])
        angle_axis = pin.log3(quat_pin.toRotationMatrix())
        return angle_axis

    def inverse(self) -> SO3:
        """Compute inverse rotation."""
        # For quaternion [x, y, z, w], inverse is [-x, -y, -z, w]
        quat_inv = self.quat.copy()
        quat_inv[:3] = -quat_inv[:3]
        return SO3(quat=quat_inv)

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Rotate a 3D point.
        
        Args:
            target: 3D point.
            
        Returns:
            Rotated point.
        """
        assert target.shape == (3,)
        return self.as_matrix() @ target

    def multiply(self, other: SO3) -> SO3:
        """Compose two rotations.
        
        Args:
            other: Another SO3 rotation.
            
        Returns:
            Composed rotation.
        """
        q1 = pin.Quaternion(self.quat[3], self.quat[0], self.quat[1], self.quat[2])
        q2 = pin.Quaternion(other.quat[3], other.quat[0], other.quat[1], other.quat[2])
        result = (q1 * q2).coeffs()  # [x, y, z, w]
        return SO3(quat=result)

    def adjoint(self) -> np.ndarray:
        """Compute adjoint matrix (same as rotation matrix for SO(3))."""
        return self.as_matrix()
