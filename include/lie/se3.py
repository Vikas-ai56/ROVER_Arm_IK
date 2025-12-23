"""SE(3): Special Euclidean group for rigid transforms in 3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .base import MatrixLieGroup
from .so3 import SO3
from .utils import get_epsilon, skew


@dataclass(frozen=True)
class SE3(MatrixLieGroup):
    """Special Euclidean group for proper rigid transforms in 3D.

    Internal parameterization uses rotation (quaternion) + translation.
    Tangent parameterization is (vx, vy, vz, omega_x, omega_y, omega_z).
    """

    rotation: SO3
    translation: np.ndarray
    matrix_dim: int = 4
    parameters_dim: int = 7
    tangent_dim: int = 6
    space_dim: int = 3

    def __repr__(self) -> str:
        rot = np.round(self.rotation.quat, 5)
        trans = np.round(self.translation, 5)
        return f"{self.__class__.__name__}(quat={rot}, xyz={trans})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SE3):
            return NotImplemented
        return self.rotation == other.rotation and np.allclose(self.translation, other.translation)

    def copy(self) -> SE3:
        return SE3(rotation=self.rotation.copy(), translation=self.translation.copy())

    def parameters(self) -> np.ndarray:
        """Return [quat_x, quat_y, quat_z, quat_w, x, y, z]."""
        return np.concatenate([self.rotation.quat, self.translation])

    @classmethod
    def identity(cls) -> SE3:
        return SE3(rotation=SO3.identity(), translation=np.zeros(3))

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: np.ndarray,
    ) -> SE3:
        """Create SE3 from rotation and translation.
        
        Args:
            rotation: SO3 rotation.
            translation: 3D translation vector.
            
        Returns:
            SE3 instance.
        """
        assert translation.shape == (3,)
        return SE3(rotation=rotation, translation=translation.copy())

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SE3:
        """Create SE3 from 4x4 homogeneous matrix.
        
        Args:
            matrix: 4x4 transformation matrix.
            
        Returns:
            SE3 instance.
        """
        assert matrix.shape == (4, 4)
        rotation = SO3.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return SE3(rotation=rotation, translation=translation)

    @classmethod
    def from_pinocchio_se3(cls, placement: pin.SE3) -> SE3:
        """Create SE3 from Pinocchio SE3 object.
        
        Args:
            placement: Pinocchio SE3 placement.
            
        Returns:
            SE3 instance.
        """
        rotation = SO3.from_matrix(placement.rotation)
        translation = placement.translation.copy()
        return SE3(rotation=rotation, translation=translation)

    @classmethod
    def from_translation(cls, translation: np.ndarray) -> SE3:
        """Create SE3 with only translation."""
        return SE3.from_rotation_and_translation(
            rotation=SO3.identity(),
            translation=translation
        )

    def as_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    @classmethod
    def exp(cls, tangent: np.ndarray) -> SE3:
        """Exponential map from tangent space to SE(3).
        
        Args:
            tangent: 6D twist [linear_vel, angular_vel].
            
        Returns:
            SE3 instance.
        """
        assert tangent.shape == (6,)
        
        v = tangent[:3]  # Linear velocity
        omega = tangent[3:]  # Angular velocity
        
        rotation = SO3.exp(omega)
        theta = np.linalg.norm(omega)
        
        if theta < get_epsilon(tangent.dtype):
            translation = v
        else:
            # Compute V matrix
            omega_skew = skew(omega)
            V = (
                np.eye(3)
                + (1 - np.cos(theta)) / (theta**2) * omega_skew
                + (theta - np.sin(theta)) / (theta**3) * (omega_skew @ omega_skew)
            )
            translation = V @ v
        
        return SE3.from_rotation_and_translation(rotation, translation)

    def log(self) -> np.ndarray:
        """Logarithm map from SE(3) to tangent space.
        
        Returns:
            6D twist [linear_vel, angular_vel].
        """
        omega = self.rotation.log()
        theta = np.linalg.norm(omega)
        
        if theta < get_epsilon(omega.dtype):
            V_inv = np.eye(3)
        else:
            omega_skew = skew(omega)
            V_inv = (
                np.eye(3)
                - 0.5 * omega_skew
                + (1 - 0.5 * theta / np.tan(0.5 * theta)) / (theta**2) * (omega_skew @ omega_skew)
            )
        
        v = V_inv @ self.translation
        return np.concatenate([v, omega])

    def inverse(self) -> SE3:
        """Compute inverse transformation."""
        rotation_inv = self.rotation.inverse()
        translation_inv = -(rotation_inv.as_matrix() @ self.translation)
        return SE3(rotation=rotation_inv, translation=translation_inv)

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Apply transformation to a 3D point.
        
        Args:
            target: 3D point.
            
        Returns:
            Transformed point.
        """
        assert target.shape == (3,)
        return self.rotation.apply(target) + self.translation

    def multiply(self, other: SE3) -> SE3:
        """Compose two transformations.
        
        Args:
            other: Another SE3 transformation.
            
        Returns:
            Composed transformation.
        """
        rotation = self.rotation.multiply(other.rotation)
        translation = self.rotation.apply(other.translation) + self.translation
        return SE3(rotation=rotation, translation=translation)

    def adjoint(self) -> np.ndarray:
        """Compute 6x6 adjoint matrix."""
        R = self.rotation.as_matrix()
        t_skew = skew(self.translation)
        
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[:3, 3:] = t_skew @ R
        adj[3:, 3:] = R
        
        return adj

    def minus(self, other: SE3) -> np.ndarray:
        """Compute the difference between two SE3 elements as a twist.
        
        This computes: log(other^{-1} * self)
        
        Args:
            other: Another SE3 transformation.
            
        Returns:
            6D twist representing the difference.
        """
        diff = other.inverse().multiply(self)
        return diff.log()
