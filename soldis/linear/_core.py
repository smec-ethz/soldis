from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Generic

import jax
from jax import Array
from jax._src.tree_util import register_pytree_node_class

from soldis.typing import JacobianT


class LinearSolverVariant(Enum):
    DIRECT = "direct"
    MATRIX_FREE = "mf"


class LinearSolver(ABC, Generic[JacobianT]):
    """Base class for linear solvers."""

    variant: ClassVar[LinearSolverVariant]

    def __init_subclass__(cls) -> None:
        """Automatically register subclasses as pytree node classes."""
        register_pytree_node_class(cls)

    def tree_flatten(self):
        aux_data = ()
        return (), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls._rebuild()

    @classmethod
    def _rebuild(cls) -> LinearSolver[JacobianT]:
        obj = cls.__new__(cls)
        return obj

    @abstractmethod
    def __call__(self, A: JacobianT, b: Array) -> Array:
        raise NotImplementedError("Subclasses must implement the __call__ method.")


class DirectLinearSolver(LinearSolver[Array]):
    """Direct linear solver for dense Jacobians."""

    variant = LinearSolverVariant.DIRECT

    def __call__(self, A: Array, b: Array) -> Array:
        return jax.numpy.linalg.solve(A, b)
