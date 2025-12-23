"""Joint velocity limit."""

from typing import Dict, List, Mapping, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class VelocityLimit(Limit):
    """Inequality constraint on joint velocities in a robot model.

    Attributes:
        indices: Tangent indices corresponding to velocity-limited joints.
        limit: Maximum allowed velocity magnitude for velocity-limited joints.
        projection_matrix: Projection from tangent space to subspace with
            velocity-limited joints.
    """

    indices: np.ndarray
    limit: np.ndarray
    projection_matrix: Optional[np.ndarray]

    def __init__(
        self,
        model: pin.Model,
        velocities: Mapping[str, npt.ArrayLike] = {},
    ):
        """Initialize velocity limits.

        Args:
            model: Pinocchio model.
            velocities: Dictionary mapping joint name to maximum allowed magnitude in
                [m]/[s] for prismatic joints and [rad]/[s] for revolute joints.
        """
        self.model = model
        limit_list: List[float] = []
        index_list: List[int] = []
        
        for joint_name, max_vel in velocities.items():
            # Find joint by name
            if not model.existJointName(joint_name):
                raise LimitDefinitionError(
                    f"Joint '{joint_name}' not found in model. "
                    f"Available joints: {[model.names[i] for i in range(len(model.names))]}"
                )
            
            joint_id = model.getJointId(joint_name)
            
            # Get velocity index for this joint
            # Note: In Pinocchio, joint indices start at 1 (0 is universe)
            if joint_id == 0:
                raise LimitDefinitionError(f"Cannot set limits for universe joint")
            
            # Get the velocity index range for this joint
            idx_v = model.joints[joint_id].idx_v
            nv_joint = model.joints[joint_id].nv
            
            max_vel = np.atleast_1d(max_vel)
            if max_vel.shape[0] == 1:
                # Broadcast to all DOFs of the joint
                max_vel = np.full(nv_joint, max_vel[0])
            elif max_vel.shape[0] != nv_joint:
                raise LimitDefinitionError(
                    f"Joint '{joint_name}' has {nv_joint} DOFs but velocity limit has "
                    f"shape {max_vel.shape}"
                )
            
            for i in range(nv_joint):
                index_list.append(idx_v + i)
                limit_list.append(max_vel[i])
        
        self.indices = np.array(index_list)
        self.limit = np.array(limit_list)
        
        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute velocity limits as QP inequalities.

        The limits are:
            -v_max <= dq <= v_max

        Args:
            configuration: Robot configuration q.
            dt: Integration timestep in [s].

        Returns:
            Constraint (G, h) or empty Constraint if there are no limits.
        """
        del dt  # Unused for velocity limits
        
        if self.projection_matrix is None:
            return Constraint()

        # Build inequality constraints: G * dq <= h
        # Upper: I * dq <= v_max
        # Lower: -I * dq <= v_max  (equivalent to dq >= -v_max)
        G_upper = self.projection_matrix
        h_upper = self.limit
        
        G_lower = -self.projection_matrix
        h_lower = self.limit
        
        G = np.vstack([G_upper, G_lower])
        h = np.concatenate([h_upper, h_lower])
        
        return Constraint(G, h)
