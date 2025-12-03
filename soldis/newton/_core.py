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

from soldis.linear._core import DirectLinearSolver, LinearSolver, LinearSolverVariant
from soldis.typing import Args, Fn, JacobianFunc, JacobianT, Y


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


def jvp_for(func: Fn[Y, Args]) -> Callable[[Y, Args], Callable[[Y], Array]]:
    def jvp(primal: Y, args: Args) -> Callable[[Y], Array]:
        """Returns a function that computes the Jacobian-vector product at `primal`."""

        def mv(v: Y) -> Array:
            _, jvp_out = jax.jvp(lambda x: func(x, args), (primal,), (v,))
            return jvp_out

        return mv

    return jvp


class Linearization(Generic[Y, Args, JacobianT]):
    linear_solver: LinearSolver[JacobianT]
    """Callable that takes (A, b) and returns the solution x to Ax = b. Where A is either
    a matrix or a function representing a matrix-vector product, depending on the type of
    Linearization."""
    jac: JacobianFunc[Y, Args, JacobianT]
    """Callable that takes (y, args) and returns the Jacobian matrix or matrix-vector
    product function."""

    def __init__(
        self,
        linear_solver: LinearSolver[JacobianT],
        jacobian_fn: JacobianFunc[Y, Args, JacobianT],
    ) -> None:
        self.linear_solver = linear_solver
        self.jac = jacobian_fn

    def compute_increment(self, y: Y, args: Args, b: Array) -> Array:
        A = self.jac(y, args)
        return self.linear_solver(A, b)


class _Solver(ABC, Generic[SolverOptionsT, Y, Args, JacobianT]):
    options: SolverOptionsT
    fn: Fn[Y, Args]
    linearization: Linearization[Y, Args, JacobianT]

    def __init_subclass__(cls) -> None:
        """Automatically register subclasses as pytree node classes."""
        register_pytree_node_class(cls)

    def tree_flatten(self):
        aux_data = (self.fn, self.linearization, self.options)
        return (), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fn, lin_op, options = aux_data
        return cls._rebuild(fn, lin_op, options)

    @classmethod
    def _rebuild(
        cls,
        fn: Fn[Y, Args],
        linearization: Linearization[Y, Args, JacobianT],
        options: SolverOptionsT,
    ) -> "_Solver[SolverOptionsT, Y, Args, JacobianT]":
        obj = cls.__new__(cls)
        obj.fn = fn
        obj.linearization = linearization
        obj.options = options
        return obj

    def __init__(
        self,
        fn: Fn[Y, Args],
        lin_solver: LinearSolver[JacobianT] | None = None,
        jac: Callable[[Y, Args], JacobianT] | None = None,
    ) -> None:
        if lin_solver is None:
            _lin_solver = DirectLinearSolver()
        else:
            _lin_solver = lin_solver

        self.fn = fn

        if jac is not None:
            self.linearization = Linearization(_lin_solver, jac)
        else:
            # for matrix-free, use jvp
            if _lin_solver.variant == LinearSolverVariant.MATRIX_FREE:
                jac_fn = jvp_for(fn)
            else:  # for direct, use jacfwd
                jac_fn = jax.jacfwd(fn)

            # construct linearization and done
            self.linearization = Linearization(_lin_solver, jac_fn)

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
