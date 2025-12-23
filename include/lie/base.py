"""Base class for matrix Lie groups."""

import abc
from typing import Union, overload

import numpy as np
from typing_extensions import Self


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups.

    Attributes:
        matrix_dim: Dimension of square matrix output.
        parameters_dim: Dimension of underlying parameters.
        tangent_dim: Dimension of tangent space.
        space_dim: Dimension of coordinates that can be transformed.
    """

    matrix_dim: int
    parameters_dim: int
    tangent_dim: int
    space_dim: int

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...

    def __matmul__(self, other: Union[Self, np.ndarray]) -> Union[Self, np.ndarray]:
        """Overload of the @ operator."""
        if isinstance(other, np.ndarray):
            return self.apply(target=other)
        assert isinstance(other, MatrixLieGroup)
        return self.multiply(other=other)

    # Factory methods

    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """Returns identity element."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: np.ndarray) -> Self:
        """Get group member from matrix representation."""
        raise NotImplementedError

    # Accessors

    @abc.abstractmethod
    def as_matrix(self) -> np.ndarray:
        """Get transformation as a matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Get underlying representation."""
        raise NotImplementedError

    # Operations

    @abc.abstractmethod
    def apply(self, target: np.ndarray) -> np.ndarray:
        """Applies group action to a point."""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: np.ndarray) -> Self:
        """Computes expm(wedge(tangent))."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self) -> np.ndarray:
        """Computes vee(logm(transformation matrix))."""
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self) -> np.ndarray:
        """Computes the adjoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse of the transform."""
        raise NotImplementedError

    def copy(self) -> Self:
        """Create a copy of this element."""
        raise NotImplementedError
