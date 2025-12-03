from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Concatenate, Generic, NamedTuple, TypeVar, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.tree_util import register_pytree_node_class

from soldis.linear._core import DirectLinearSolver, LinearSolver, LinearSolverVariant
from soldis.typing import ArgsTuple, Fn, JacobianFunc, JacobianT, Mv, P, Y


class SolverState(NamedTuple, Generic[Y, P]):
    value: Y
    args: ArgsTuple  # Unpack[P]
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
    func: Fn[Y, P],
) -> Callable[Concatenate[Y, P], Callable[[Y], Array]]:
    def jvp(primal: Y, *args: P.args) -> Callable[[Y], Array]:
        """Returns a function that computes the Jacobian-vector product at `primal`."""

        def mv(v: Y) -> Array:
            _, jvp_out = jax.jvp(lambda x: func(x, *args), (primal,), (v,))
            return jvp_out

        return mv

    return jvp


class _Solver(ABC, Generic[SolverOptionsT, Y, P, JacobianT]):
    fn: Fn[Y, P]
    """The function for which to find the root. Usually, this is the residual."""

    linear_solver: LinearSolver[JacobianT]
    """Callable that takes (A, b) and returns the solution x to Ax = b. Where A is either
    a matrix or a function representing a matrix-vector product, depending on the type of
    Linearization."""

    jac: JacobianFunc[Y, P, JacobianT]
    """Callable that takes (y, *args) and returns the Jacobian matrix or matrix-vector
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
        return cls(fn, linear_solver, jac, options=options)

    @overload  # direct matrix branch
    def __init__(
        self: _Solver[SolverOptionsT, Y, P, Array],
        fn: Fn[Y, P],
        lin_solver: None = None,
        jac: None = None,
        *,
        options: SolverOptionsT | None = None,
        **kwargs,
    ) -> None: ...
    @overload  # matrix-free branch
    def __init__(
        self: _Solver[SolverOptionsT, Y, P, JacobianT],
        fn: Fn[Y, P],
        lin_solver: LinearSolver[JacobianT],
        jac: Callable[Concatenate[Y, P], JacobianT] | None = None,
        *,
        options: SolverOptionsT | None = None,
        **kwargs,
    ) -> None: ...
    def __init__(
        self,
        fn,
        lin_solver=None,
        jac=None,
        *,
        options=None,
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

    def root(self, y0: Y, *args: P.args) -> SolverState[Y, P]:
        state = self.init(y0, *args)

        def cond_fn(state: SolverState[Y, P]) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: SolverState[Y, P]) -> SolverState[Y, P]:
            return self.step(state)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state

    def compute_increment(self, y: Y, args: ArgsTuple, b: Array) -> Array:
        A = self.jac(y, *args)
        return self.linear_solver(A, b)

    @abstractmethod
    def init(self, y0: Y, *args: P.args) -> SolverState[Y, P]: ...

    @abstractmethod
    def step(self, state: SolverState[Y, P]) -> SolverState[Y, P]: ...

    @abstractmethod
    def terminate(self, state: SolverState[Y, P]) -> Array: ...

    def _make_default_options(self, **kwargs) -> SolverOptionsT:
        """Construct default options. Override in subclasses for custom options."""
        return SolverOptions(**kwargs)  # type: ignore[return-value]
