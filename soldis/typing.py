from collections.abc import Callable
from typing import Any, Concatenate

from jax import Array
from typing_extensions import TypeVar  # needed for python < 3.13

type Fn[Y: Array, **P] = Callable[Concatenate[Y, P], Array]
type Mv = Callable[[Array], Array]  # Matrix-vector product function

type Jacobian = Array | Mv | Any
JacobianT = TypeVar("JacobianT", bound=Jacobian, default=Array)

type JacobianFunc[Y: Array, **P, JacobianT] = Callable[Concatenate[Y, P], JacobianT]
type LinearSolve = Callable[[JacobianT, Array], Array]
type Y = Array
