"""Exceptions specific to the IK solver."""

import pinocchio as pin


class IKError(Exception):
    """Base class for IK solver exceptions."""


class NoSolutionFound(IKError):
    """Exception raised when the QP solver fails to find a solution."""

    def __init__(self, solver_name: str):
        super().__init__(f"QP solver {solver_name} failed to find a solution.")


class InvalidFrame(IKError):
    """Exception raised when a frame name is not found in the robot model."""

    def __init__(self, frame_name: str, model: pin.Model):
        available_frames = [model.frames[i].name for i in range(len(model.frames))]
        message = (
            f"Frame '{frame_name}' does not exist in the model. "
            f"Available frame names: {available_frames}"
        )
        super().__init__(message)


class NotWithinConfigurationLimits(IKError):
    """Exception raised when a configuration violates its limits."""

    def __init__(self, joint_id: int, value: float, lower: float, upper: float, model: pin.Model):
        joint_name = model.names[joint_id] if joint_id < len(model.names) else f"joint_{joint_id}"
        message = (
            f"Configuration violates limits for joint '{joint_name}' (id={joint_id}). "
            f"Value: {value:.4f}, Limits: [{lower:.4f}, {upper:.4f}]"
        )
        super().__init__(message)


class TargetNotSet(IKError):
    """Exception raised when a task target has not been set."""

    def __init__(self, task_name: str):
        super().__init__(f"Target not set for task: {task_name}")


class InvalidTarget(IKError):
    """Exception raised when a task target is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidGain(IKError):
    """Exception raised when a task gain is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidDamping(IKError):
    """Exception raised when damping parameter is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class TaskDefinitionError(IKError):
    """Exception raised when a task is incorrectly defined."""

    def __init__(self, message: str):
        super().__init__(message)


class LimitDefinitionError(IKError):
    """Exception raised when a limit is incorrectly defined."""

    def __init__(self, message: str):
        super().__init__(message)
