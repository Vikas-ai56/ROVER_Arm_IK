"""Build and solve the inverse kinematics problem.

This module provides functions to formulate and solve the IK problem as a
quadratic program (QP). The QP minimizes weighted task errors subject to
joint limit constraints.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import osqp
from scipy import sparse

from .configuration import Configuration
from .exceptions import NoSolutionFound
from .limits import ConfigurationLimit, Constraint, Limit
from .tasks import BaseTask, Objective


def _compute_qp_objective(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    damping: float,
) -> Objective:
    """Compute the QP objective from all tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        damping: Levenberg-Marquardt damping for numerical stability.

    Returns:
        Combined objective (H, c) from all tasks.
    """
    # Initialize with damping regularization
    H = np.eye(configuration.nv) * damping
    c = np.zeros(configuration.nv)

    # Accumulate objectives from all tasks
    for task in tasks:
        obj = task.compute_qp_objective(configuration)
        H_task, c_task = obj.H, obj.c
        H += H_task
        c += c_task

    return Objective(H, c)

def _compute_qp_inequalities(
    configuration: Configuration,
    limits: Optional[Sequence[Limit]],
    dt: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute the QP inequality constraints from all limits.

    Args:
        configuration: Robot configuration.
        limits: List of limits to enforce.
        dt: Integration timestep in [s].

    Returns:
        Tuple (G, h) for inequality constraints G @ dq <= h, or (None, None)
        if no active constraints.
    """
    if limits is None:
        # Default to configuration limits
        limits = [ConfigurationLimit(configuration.model)]

    G_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []

    for limit in limits:
        constraint = limit.compute_qp_inequalities(configuration, dt)
        if not constraint.inactive:
            assert constraint.G is not None and constraint.h is not None
            G_list.append(constraint.G)
            h_list.append(constraint.h)

    if not G_list:
        return None, None

    return np.vstack(G_list), np.hstack(h_list)


def build_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Build the quadratic program for inverse kinematics.

    The quadratic program is defined as:
        minimize:   (1/2) * dq^T * H * dq + c^T * dq
        subject to: G * dq <= h

    where dq = v * dt is the vector of joint displacements.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping. Higher values improve numerical
            stability but slow down task convergence.
        limits: List of limits to enforce. Set to empty list to disable all limits.
            If None, defaults to configuration limits only.

    Returns:
        Tuple (H, c, G, h) representing the QP.
    """
    H, c = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    return H, c, G, h


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
    solver_verbose: bool = False,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
) -> np.ndarray:
    """Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies (at weighted best) the set of provided kinematic tasks
    while respecting joint limits.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping applied to all tasks. Higher values
            improve numerical stability but slow down task convergence.
        safety_break: If True, stop execution and raise an exception if the
            current configuration is outside limits. If False, log a warning
            and continue execution.
        limits: List of limits to enforce. Set to empty list to disable all limits.
            If None, defaults to configuration limits only.
        solver_verbose: If True, print OSQP solver output.
        eps_abs: Absolute tolerance for OSQP solver.
        eps_rel: Relative tolerance for OSQP solver.

    Returns:
        Velocity v in tangent space of shape (nv,).

    Raises:
        NotWithinConfigurationLimits: If the current configuration is outside
            the joint limits and safety_break is True.
        NoSolutionFound: If the QP solver fails to find a solution.

    Example:
        >>> config = Configuration.from_urdf("robot.urdf")
        >>> frame_task = FrameTask("end_effector", position_cost=1.0, orientation_cost=1.0)
        >>> frame_task.set_target(SE3.from_translation(np.array([0.5, 0.2, 0.3])))
        >>> velocity = solve_ik(config, [frame_task], dt=0.01)
        >>> config.integrate_inplace(velocity, dt=0.01)
    """
    # Check that current configuration is within limits
    configuration.check_limits(safety_break=safety_break)

    # Build the QP
    H, c, G, h = build_ik(configuration, tasks, dt, damping, limits)

    # Convert to sparse format for OSQP
    P = sparse.csr_matrix(H)
    q = c

    # Set up OSQP solver
    solver = osqp.OSQP()
    
    if G is not None and h is not None:
        A = sparse.csr_matrix(G)
        u = h
        l = -np.inf * np.ones_like(h)
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=solver_verbose,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )
    else:
        # No constraints
        solver.setup(
            P=P,
            q=q,
            verbose=solver_verbose,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )

    # Solve
    result = solver.solve()

    if result.info.status != "solved":
        raise NoSolutionFound(f"OSQP (status: {result.info.status})")

    delta_q = result.x

    # Convert displacement to velocity
    v: np.ndarray = delta_q / dt
    return v
