"""All kinematic limits derive from the Limit base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    """Linear inequality constraint of the form G(q) * dq <= h(q).

    Inactive if G and h are None.
    """

    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None

    @property
    def inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return self.G is None and self.h is None


class Limit(abc.ABC):
    """Abstract base class for kinematic limits.

    Subclasses must implement the compute_qp_inequalities method
    which takes in the current robot configuration and integration time step and
    returns an instance of Constraint.
    """

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute limit as linearized QP inequalities of the form:

            G(q) * dq <= h(q)

        where q is the robot's configuration and dq is the displacement in the 
        tangent space at q.

        Args:
            configuration: Robot configuration q.
            dt: Integration time step in [s].

        Returns:
            Constraint (G, h).
        """
        raise NotImplementedError
