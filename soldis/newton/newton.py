from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import jax
import jax.numpy as jnp

from soldis.newton._core import SolverOptions, SolverState, _Solver
from soldis.typing import Array, JacobianT, P, Y


@dataclass(frozen=True)
class NewtonSolverOptions(SolverOptions):
    norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm


class NewtonSolver(_Solver[NewtonSolverOptions, Y, P, JacobianT]):
    """Dense Newton-Raphson solver implementation."""

    def _make_default_options(self, **kwargs) -> NewtonSolverOptions:
        return NewtonSolverOptions(**kwargs)

    def init(
        self,
        y0: Y,
        *args: P.args,
    ) -> SolverState[Y, P]:
        initial_residual = self.fn(y0, *args)
        initial_converged = self.options.norm_fn(initial_residual) < self.options.tol

        return SolverState(
            value=y0,
            args=args,
            residual=initial_residual,
            iteration=jnp.asarray(0),
            converged=initial_converged,
        )

    def step(self, state: SolverState[Y, P]) -> SolverState[Y, P]:
        """Perform a single iteration step."""
        delta = self.compute_increment(state.value, state.args, -state.residual)
        new_value = cast(Y, state.value + delta)

        # Check convergence
        new_residual = self.fn(new_value, *state.args)
        new_converged = self.options.norm_fn(new_residual) < self.options.tol

        return state._replace(
            value=new_value,
            residual=new_residual,
            iteration=state.iteration + 1,
            converged=new_converged,
        )

    def terminate(self, state: SolverState[Y, P]) -> Array:
        """Check if the solver should terminate."""
        return jnp.logical_or(state.converged, state.iteration >= self.options.maxiter)


@dataclass(frozen=True)
class LineSearchNewtonSolverOptions(NewtonSolverOptions):
    ls_maxiter: int = 10
    ls_decrease: float = 0.5
    ls_c: float = 1e-4


class LineSearchNewtonSolver(_Solver[LineSearchNewtonSolverOptions, Y, P, JacobianT]):
    """Newton-Raphson solver with backtracking line search."""

    def _make_default_options(self, **kwargs) -> LineSearchNewtonSolverOptions:
        return LineSearchNewtonSolverOptions(**kwargs)

    def init(
        self,
        y0: Y,
        *args: P.args,
    ) -> SolverState[Y, P]:
        initial_residual = self.fn(y0, *args)
        initial_converged = self.options.norm_fn(initial_residual) < self.options.tol

        return SolverState(
            value=y0,
            args=args,
            residual=initial_residual,
            iteration=jnp.asarray(0),
            converged=initial_converged,
        )

    def step(self, state: SolverState[Y, P]) -> SolverState[Y, P]:
        direction = self.compute_increment(state.value, state.args, -state.residual)
        current_norm = self.options.norm_fn(state.residual)

        def cond_fn(carry):
            _, ls_iter, accepted, _, _, _ = carry
            return jnp.logical_and(
                ls_iter < self.options.ls_maxiter, jnp.logical_not(accepted)
            )

        def body_fn(carry):
            step_size, ls_iter, _, _, _, _ = carry
            candidate_value = cast(Y, state.value + step_size * direction)
            candidate_residual = self.fn(candidate_value, *state.args)
            candidate_norm = self.options.norm_fn(candidate_residual)
            accepted = (
                candidate_norm <= (1.0 - self.options.ls_c * step_size) * current_norm
            )
            next_step = jnp.where(
                accepted, step_size, step_size * self.options.ls_decrease
            )
            return (
                next_step,
                ls_iter + 1,
                accepted,
                candidate_value,
                candidate_residual,
                candidate_norm,
            )

        init_carry = (
            jnp.asarray(1.0),
            jnp.asarray(0),
            jnp.asarray(False),
            state.value,
            state.residual,
            current_norm,
        )

        step_size, _, _, new_value, new_residual, new_norm = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )

        new_converged = new_norm < self.options.tol

        return state._replace(
            value=new_value,
            residual=new_residual,
            iteration=state.iteration + 1,
            converged=new_converged,
        )

    def terminate(self, state: SolverState[Y, P]) -> Array:
        return jnp.logical_or(state.converged, state.iteration >= self.options.maxiter)
