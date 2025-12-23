"""Joint position limit."""

from typing import List

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class ConfigurationLimit(Limit):
    """Inequality constraint on joint positions in a robot model.

    This enforces that after integration, the configuration remains within bounds:
        q_min <= q + dq*dt <= q_max
    """

    def __init__(
        self,
        model: pin.Model,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        """Initialize configuration limits.

        Args:
            model: Pinocchio model.
            gain: Gain factor in (0, 1] that determines how fast each joint is
                allowed to move towards the joint limits at each timestep. Values lower
                than 1 are safer but may make the joints move slowly.
            min_distance_from_limits: Offset in meters (prismatic joints) or radians
                (revolute joints) to be added to the limits. Positive values decrease the
                range of motion, negative values increase it.
        """
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        self.model = model
        self.gain = gain
        
        # Get joint limits from model
        self.lower = model.lowerPositionLimit.copy() + min_distance_from_limits
        self.upper = model.upperPositionLimit.copy() - min_distance_from_limits
        
        # Find which joints have limits
        index_list: List[int] = []
        for i in range(model.nv):
            # Check if this DOF has finite limits
            if np.isfinite(self.lower[i]) and np.isfinite(self.upper[i]):
                index_list.append(i)
        
        self.indices = np.array(index_list)
        self.indices.setflags(write=False)
        
        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float = 1.0,
    ) -> Constraint:
        """Compute the configuration-dependent joint position limits.

        The limits are defined as:
            (q_min - q) / dt <= dq <= (q_max - q) / dt

        This ensures that after integration q + dq*dt stays within [q_min, q_max].

        Args:
            configuration: Robot configuration q.
            dt: Integration timestep in [s].

        Returns:
            Constraint (G, h) representing the inequality constraint as
            G * dq <= h, or empty Constraint if there are no limits.
        """
        if self.projection_matrix is None:
            return Constraint()

        q = configuration.q
        
        # Compute differences using Pinocchio's difference function
        # Upper limit: dq <= (q_max - q) / dt
        delta_q_max = pin.difference(self.model, q, self.upper)
        
        # Lower limit: -dq <= -(q_min - q) / dt  or  dq >= (q_min - q) / dt
        delta_q_min = pin.difference(self.model, self.lower, q)
        
        # Apply gain to slow down approach to limits
        delta_q_max_safe = self.gain * delta_q_max / dt
        delta_q_min_safe = -self.gain * delta_q_min / dt
        
        # Build inequality constraints: G * dq <= h
        # Upper: I * dq <= delta_q_max_safe
        # Lower: -I * dq <= -delta_q_min_safe
        G_upper = self.projection_matrix
        h_upper = delta_q_max_safe[self.indices]
        
        G_lower = -self.projection_matrix
        h_lower = -delta_q_min_safe[self.indices]
        
        G = np.vstack([G_upper, G_lower])
        h = np.concatenate([h_upper, h_lower])
        
        return Constraint(G, h)
