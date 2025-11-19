from typing import (
    Callable,
    TypeAlias,
    TypeVar,
)

from jax import Array

Y = TypeVar("Y", bound=Array)  # could be a pytree, but likely a jax.Array
Args = TypeVar("Args")

Fn: TypeAlias = Callable[[Y, Args], Array]
Mv: TypeAlias = Callable[[Array], Array]  # Matrix-vector product function
Jacobian: TypeAlias = Array | Mv
JacobianT = TypeVar("JacobianT", bound=Jacobian)
LinearSolver: TypeAlias = Callable[[JacobianT, Array], Array]
