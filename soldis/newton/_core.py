from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, NamedTuple, TypeVar, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.tree_util import register_pytree_node_class

from soldis.linear._core import DirectLinearSolver, LinearSolver, LinearSolverVariant
from soldis.typing import Args, Fn, JacobianFunc, JacobianT, Mv, Y


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


def _default_jvp_factory(
    func: Fn[Y, Args],
) -> Callable[[Y, Args], Callable[[Y], Array]]:
    def jvp(primal: Y, args: Args) -> Callable[[Y], Array]:
        """Returns a function that computes the Jacobian-vector product at `primal`."""

        def mv(v: Y) -> Array:
            _, jvp_out = jax.jvp(lambda x: func(x, args), (primal,), (v,))
            return jvp_out

        return mv

    return jvp


class _Solver(ABC, Generic[SolverOptionsT, Y, Args, JacobianT]):
    fn: Fn[Y, Args]
    """The function for which to find the root. Usually, this is the residual."""

    linear_solver: LinearSolver[JacobianT]
    """Callable that takes (A, b) and returns the solution x to Ax = b. Where A is either
    a matrix or a function representing a matrix-vector product, depending on the type of
    Linearization."""

    jac: JacobianFunc[Y, Args, JacobianT]
    """Callable that takes (y, args) and returns the Jacobian matrix or matrix-vector
    product function."""

    options: SolverOptionsT
    """Solver options. Subclasses may define custom options by extending SolverOptions."""

    def __init_subclass__(cls) -> None:
        """Automatically register subclasses as pytree node classes."""
        register_pytree_node_class(cls)

    def tree_flatten(self):
        aux_data = (self.fn, self.linear_solver, self.jac, self.options)
        return (), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fn, linear_solver, jac, options = aux_data
        return cls(fn, linear_solver, jac, options)

    @overload  # direct matrix branch
    def __init__(
        self: _Solver[SolverOptionsT, Y, Args, Array],
        fn: Fn[Y, Args],
        lin_solver: LinearSolver[Array] | None = None,
        jac: Callable[[Y, Args], Array] | None = None,
        *,
        options: SolverOptionsT | None = None,
        **kwargs,
    ) -> None: ...
    @overload  # matrix-free branch
    def __init__(
        self: _Solver[SolverOptionsT, Y, Args, Mv],
        fn: Fn[Y, Args],
        lin_solver: LinearSolver[Mv],
        jac: Callable[[Y, Args], Mv] | None = None,
        *,
        options: SolverOptionsT | None = None,
        **kwargs,
    ) -> None: ...
    def __init__(
        self,
        fn: Fn[Y, Args],
        lin_solver: LinearSolver[JacobianT] | None = None,
        jac: Callable[[Y, Args], JacobianT] | None = None,
        *,
        options: SolverOptionsT | None = None,
        **kwargs,
    ) -> None:
        self.fn = fn

        if lin_solver is None:
            _lin_solver = DirectLinearSolver()
        else:
            _lin_solver = lin_solver
        self.linear_solver = _lin_solver  # ty: ignore[invalid-assignment]

        if jac is not None:
            self.jac = jac
        else:
            # for matrix-free, use jvp
            if _lin_solver.variant == LinearSolverVariant.MATRIX_FREE:
                jac_fn = _default_jvp_factory(fn)
            else:  # for direct, use jacfwd
                jac_fn = jax.jacfwd(fn)

            # construct linearization and done
            self.jac = jac_fn  # ty: ignore[invalid-assignment]

        # handle options
        if options is None:
            self.options = self._make_default_options(**kwargs)
        elif kwargs:
            raise TypeError(
                "Pass either an options instance or option keyword arguments, not both"
            )
        else:
            self.options = options

    def root(self, y0: Y, args: Args) -> SolverState[Y, Args]:
        state = self.init(y0, args)

        def cond_fn(state: SolverState) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: SolverState[Y, Args]) -> SolverState[Y, Args]:
            return self.step(state)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state

    def compute_increment(self, y: Y, args: Args, b: Array) -> Array:
        A = self.jac(y, args)
        return self.linear_solver(A, b)

    @abstractmethod
    def init(self, y0: Y, args: Args) -> SolverState[Y, Args]: ...

    @abstractmethod
    def step(self, state: SolverState) -> SolverState: ...

    @abstractmethod
    def terminate(self, state: SolverState) -> Array: ...

    def _make_default_options(self, **kwargs) -> SolverOptionsT:
        """Construct default options. Override in subclasses for custom options."""
        return SolverOptions(**kwargs)  # type: ignore[return-value]
