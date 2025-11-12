"""Basic Newton Raphson solver and derivatives."""

from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar

import jax
import jax.numpy as jnp


class SolverState(Protocol):
    """Class to hold the state of the solver."""

    def get_residual(self) -> jax.Array:
        """Get the current residual."""
        ...

    def update(self, other: SolverState) -> SolverState:
        """Add two solver states."""
        ...


T = TypeVar(
    "T", bound=SolverState
)  # Needs to be a pytree, or more general, a jax jit compatible type
_NRState: TypeAlias = tuple[int, T, float, bool]


class LinearSolver(Protocol):
    """Abstract base class for linear solvers used in Newton Raphson."""

    def solve(self, A: jax.Array, b: jax.Array) -> jax.Array:
        """Solve the linear system Ax = b.

        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.

        Returns:
            Solution vector x.
        """
        ...


def newton_raphson(
    initial_state: T,
    linear_solver: LinearSolver,
    *,
    maxit: int = 50,
    tol: float = 1e-10,
    verbose: bool = False,
) -> T:
    """Basic Newton Raphson solver.

    Args:
        initial_state: Initial guess for the solution.
        maxit: Maximum number of iterations.
        tol: Tolerance for convergence.
        verbose: If True, prints iteration details.

    Returns:
        The solution as a tuple of the same type as initial_state.
    """

    def cond_fun(state: _NRState[T]) -> bool:
        """Condition function for the while loop."""
        iteration, _, residual, converged = state
        if verbose:
            jax.debug.print(
                " " * 4 + "Iteration {}/{}: Residual = {}", iteration, maxit, residual
            )
        cond = (iteration < maxit) & (not converged)
        return cond  # ty: ignore[invalid-return-type]

    def body_fun(state: _NRState[T]) -> _NRState[T]:
        """Body function for the while loop."""
        iteration, current_state, _, _ = state

        # Compute residual and Jacobian
        res = current_state.get_residual()
        jacobian = current_state.get_jacobian()

        # Solve for the update
        delta = linear_solver.solve(jacobian, -res)

        # Update the state
        new_state = current_state.update(delta)

        # Compute new residual norm
        res = new_state.get_residual()
        res_norm = jnp.linalg.norm(res)
        converged = res_norm < tol

        return (iteration + 1, new_state, res_norm, converged)

    state: _NRState[T] = (0, initial_state, jnp.inf, False)
    res = jax.lax.while_loop(cond_fun, body_fun, state)
    return res[1]
