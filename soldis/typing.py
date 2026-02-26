from typing import Callable, Concatenate, ParamSpec, TypeAlias, TypeVar

from jax import Array
from tatva.sparse import ColoredMatrix

Y = TypeVar("Y", bound=Array)  # could be a pytree, but likely a jax.Array
P = ParamSpec("P")

Fn: TypeAlias = Callable[Concatenate[Y, P], Array]
Mv: TypeAlias = Callable[[Array], Array]  # Matrix-vector product function

Jacobian: TypeAlias = Array | Mv | ColoredMatrix
JacobianT = TypeVar("JacobianT", bound=Jacobian, default=Array)
JacobianFunc: TypeAlias = Callable[Concatenate[Y, P], JacobianT]

LinearSolve: TypeAlias = Callable[[JacobianT, Array], Array]
