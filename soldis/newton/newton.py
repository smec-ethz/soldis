from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    cast,
    overload,
)

import jax
import jax.numpy as jnp

from soldis.newton._core import SolverOptions, SolverState, _Solver
from soldis.typing import Args, Array, Fn, JacobianT, LinearSolve, Mv, Y


@dataclass(frozen=True)
class NewtonSolverOptions(SolverOptions):
    norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm


class NewtonSolver(_Solver[NewtonSolverOptions, Y, Args, JacobianT]):
    """Dense Newton-Raphson solver implementation."""

    @overload  # direct matrix branch
    def __init__(
        self: NewtonSolver[Y, Args, Array],
        fn: Fn[Y, Args],
        lin_solver: LinearSolve[Array] | None = None,
        jac: Callable[[Y, Args], Array] | None = None,
        *,
        maxiter: int = 50,
        tol: float = 1e-10,
        verbose: bool = False,
        norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm,
    ) -> None: ...
    @overload  # matrix-free branch
    def __init__(
        self: NewtonSolver[Y, Args, Mv],
        fn: Fn[Y, Args],
        lin_solver: LinearSolve[Mv],
        jac: Callable[[Y, Args], Mv] | None = None,
        *,
        maxiter: int = 50,
        tol: float = 1e-10,
        verbose: bool = False,
        norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm,
    ) -> None: ...
    def __init__(
        self,
        fn,
        lin_solver=None,
        jac=None,
        *,
        maxiter=50,
        tol=1e-10,
        verbose=False,
        norm_fn=jax.numpy.linalg.norm,
    ) -> None:
        # TODO: assert jac and lin_solve are compatible
        super().__init__(fn, lin_solver, jac)
        self.options = NewtonSolverOptions(
            maxiter=maxiter, tol=tol, verbose=verbose, norm_fn=norm_fn
        )

    def init(
        self,
        y0: Y,
        args: Args,
    ) -> SolverState[Y, Args]:
        initial_residual = self.fn(y0, args)
        initial_converged = self.options.norm_fn(initial_residual) < self.options.tol

        return SolverState(
            value=y0,
            args=args,
            residual=initial_residual,
            iteration=jnp.asarray(0),
            converged=initial_converged,
        )

    def step(self, state: SolverState[Y, Args]) -> SolverState[Y, Args]:
        """Perform a single iteration step."""
        delta = self.linearization.compute_increment(
            state.value, state.args, -state.residual
        )
        new_value = cast(Y, state.value + delta)

        # Check convergence
        new_residual = self.fn(new_value, state.args)
        new_converged = self.options.norm_fn(new_residual) < self.options.tol

        return state._replace(
            value=new_value,
            residual=new_residual,
            iteration=state.iteration + 1,
            converged=new_converged,
        )

    def terminate(self, state: SolverState[Y, Args]) -> Array:
        """Check if the solver should terminate."""
        return jnp.logical_or(state.converged, state.iteration >= self.options.maxiter)
