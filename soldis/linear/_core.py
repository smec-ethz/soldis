from typing import (
    Callable,
    Generic,
    NamedTuple,
    Self,
)

import jax
from jax import Array

from soldis.typing import Args, Fn, JacobianT, LinearSolver, Y


class LinearOperator(NamedTuple, Generic[Y, Args, JacobianT]):
    solve_fn: LinearSolver[JacobianT]
    fn: Fn[Y, Args]
    jac: Callable[[Y, Args], JacobianT]

    @classmethod
    def from_fn(cls, solve_fn: LinearSolver[JacobianT], fn: Fn[Y, Args]) -> Self:
        jac = jax.jacfwd(fn)
        return cls(solve_fn, fn, jac)


def compute_increment(lin_op: LinearOperator, args: tuple[Y, Args], b: Array) -> Array:
    A = lin_op.jac(*args)
    return lin_op.solve_fn(A, b)


# -----------------------------------------------
# Basic linear solver implementations
# -----------------------------------------------
def direct(A: Array, b: Array) -> Array:
    """Direct linear solver using jax.numpy.linalg.solve."""
    return jax.numpy.linalg.solve(A, b)
