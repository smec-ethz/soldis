from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    NamedTuple,
    TypeVar,
)

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.tree_util import register_pytree_node_class

from soldis.linear import LinearOperator
from soldis.typing import Args, Fn, JacobianT, LinearSolver, Y


class SolverState(NamedTuple, Generic[Y, Args]):
    value: Y
    args: Args
    residual: Array
    iteration: Array  # int
    converged: Array  # bool


@dataclass(frozen=True)
class SolverOptions:
    maxiter: int = 50
    tol: float = 1e-10
    verbose: bool = False


SolverOptionsT = TypeVar("SolverOptionsT", bound=SolverOptions)


@register_pytree_node_class
class _Solver(ABC, Generic[SolverOptionsT, Y, Args, JacobianT]):
    options: SolverOptionsT
    fn: Fn[Y, Args]
    lin_op: LinearOperator[Y, Args, JacobianT]

    def tree_flatten(self):
        children = (self.fn, self.lin_op)
        aux_data = (self.options,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (options,) = aux_data
        fn, lin_op = children
        return cls._rebuild(fn, lin_op, options)

    @classmethod
    def _rebuild(
        cls,
        fn: Fn[Y, Args],
        lin_op: LinearOperator[Y, Args, JacobianT],
        options: SolverOptionsT,
    ) -> "_Solver[SolverOptionsT, Y, Args, JacobianT]":
        obj = cls.__new__(cls)
        obj.fn = fn
        obj.lin_op = lin_op
        obj.options = options
        return obj

    def __init__(
        self,
        fn: Fn[Y, Args],
        jac: Callable[[Y, Args], JacobianT] | None = None,
        lin_solve: LinearSolver[JacobianT] | None = None,
    ) -> None:
        if lin_solve is None:
            lin_solve = jnp.linalg.solve

        if jac is None:
            jac = jax.jacfwd(fn)

        self.fn = fn
        self.lin_op = LinearOperator(lin_solve, fn, jac)

    def root(self, y0: Y, args: Args) -> SolverState[Y, Args]:
        state = self.init(y0, args)

        def cond_fn(state: SolverState) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: SolverState[Y, Args]) -> SolverState[Y, Args]:
            return self.step(state)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state

    @abstractmethod
    def init(self, y0: Y, args: Args) -> SolverState[Y, Args]: ...

    @abstractmethod
    def step(self, state: SolverState) -> SolverState: ...

    @abstractmethod
    def terminate(self, state: SolverState) -> Array: ...
