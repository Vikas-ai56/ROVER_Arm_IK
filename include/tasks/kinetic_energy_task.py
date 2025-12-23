"""Kinetic energy regularization task implementation for Pinocchio."""

from __future__ import annotations

from typing import Optional, SupportsFloat, TYPE_CHECKING

import numpy as np
import pinocchio as pin

if TYPE_CHECKING:
    from ..configuration import Configuration

from ..exceptions import TaskDefinitionError
from .task import BaseTask, Objective


class KineticEnergyRegularizationTask(BaseTask):
    """Kinetic-energy regularization.

    This task penalizes the system's kinetic energy, providing an inertia-weighted
    damping effect. It contributes the following term to the QP:

        (1/2) * λ * Δq^T * M(q) * Δq

    where Δq is the vector of joint displacements, M(q) is the joint-space
    inertia matrix, and λ is the scalar strength of the regularization.

    This can be seen as an inertia-weighted version of DampingTask.
    Degrees of freedom with higher inertia will move less for the same cost.

    Note:
        The integration timestep dt must be set via set_dt() before use.
        This ensures the cost is expressed in units of energy (Joules).

    Example:
        >>> task = KineticEnergyRegularizationTask(cost=1e-4)
        >>> task.set_dt(0.02)
    """

    def __init__(self, cost: SupportsFloat):
        """Initialize kinetic energy regularization task.
        
        Args:
            cost: Scalar cost weight (must be >= 0).
        """
        cost = float(cost)
        if cost < 0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost: float = cost

        # Kinetic energy is T = ½ * q̇^T * M * q̇
        # Since q̇ ≈ Δq / dt, we substitute:
        # T ≈ ½ * (Δq / dt)^T * M * (Δq / dt) = ½ * Δq^T * (M / dt²) * Δq
        # Therefore, we scale the inertia matrix by 1 / dt²
        self.inv_dt_sq: Optional[float] = None

    def set_dt(self, dt: float) -> None:
        """Set the integration timestep.

        Args:
            dt: Integration timestep in [s].
        """
        if dt <= 0:
            raise TaskDefinitionError("Integration timestep must be > 0")
        self.inv_dt_sq = 1.0 / dt**2

    def compute_qp_objective(self, configuration: "Configuration") -> Objective:
        """Compute the QP objective (H, c) for kinetic energy regularization.

        Args:
            configuration: Robot configuration q.

        Returns:
            Objective with H = cost * M(q) / dt² and c = 0.
        """
        if self.inv_dt_sq is None:
            raise TaskDefinitionError(
                f"{self.__class__.__name__}: Integration timestep not set. "
                "Call set_dt() before using this task."
            )
        
        # Get joint-space inertia matrix M(q)
        M = pin.crba(configuration.model, configuration.data, configuration.q)
        
        # Scale by cost and inverse timestep squared
        H = self.cost * self.inv_dt_sq * M
        
        # No linear term
        c = np.zeros(configuration.nv)
        
        return Objective(H, c)
