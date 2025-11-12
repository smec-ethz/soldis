from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, NamedTuple, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

Y = TypeVar("Y", bound=Array)  # could be a pytree, but likely a jax.Array
Args = TypeVar("Args")

Fn: TypeAlias = Callable[[Y, Args], Array]

SolverState = TypeVar("SolverState", bound=tuple)
SolverOptions = TypeVar("SolverOptions", bound=NamedTuple)


class _Solver(ABC, Generic[SolverState, Y, Args, SolverOptions]):
    """Protocol for a dense Newton-Raphson solver."""

    options: SolverOptions

    def __init__(self, fn: Fn[Y, Args], *, options: SolverOptions) -> None:
        self.fn = fn
        self.options = options

    @abstractmethod
    def init(
        self, y0: Y, args: Args, *, options: SolverOptions | None = None
    ) -> SolverState:
        """Initialize the solver state."""
        ...

    @abstractmethod
    def step(self, state: SolverState) -> SolverState:
        """Perform a single iteration step."""
        ...

    def terminate(self, state: SolverState) -> Array:
        """Check if the solver should terminate."""
        ...


class LinearOperatorState(NamedTuple, Generic[Y, Args]):
    """Container for linear solver state."""

    linsolve: Callable[[Array, Array], Array]
    fn: Fn[Y, Args]
    jacobian: Callable[[Y, Args], Array]


class NewtonState(NamedTuple, Generic[Y, Args]):
    value: Y
    args: Args
    iteration: Array  # int
    converged: Array  # bool
    linear_solver_state: LinearOperatorState[Y, Args]


class NewtonOptions(NamedTuple):
    max_iterations: int = 50
    tol: float = 1e-10
    verbose: bool = False
    linear_solver: Callable[[Array, Array], Array] | None = None
    """A callable that solves linear systems of the form Ax = b. Defaults to
    `jnp.linalg.solve`."""


class DenseNewtonSolver(_Solver[NewtonState, Y, Args, NewtonOptions]):
    """Dense Newton-Raphson solver implementation."""

    def init(
        self, y0: Y, args: Args, *, options: NewtonOptions | None = None
    ) -> NewtonState[Y, Args]:
        if options is not None:
            self.options = options

        # prepare the dense jacobian function using forward-mode autodiff
        jacobian = jax.jacfwd(self.fn)  # linear operator but dense matrix

        linear_solver_state = LinearOperatorState(
            linsolve=self.options.linear_solver or jnp.linalg.solve,
            fn=self.fn,
            jacobian=jacobian,
        )
        initial_residual = self.fn(y0, args)
        initial_converged = jax.numpy.linalg.norm(initial_residual) < self.options.tol

        return NewtonState(
            value=y0,
            args=args,
            iteration=jnp.asarray(0),
            converged=initial_converged,
            linear_solver_state=linear_solver_state,
        )

    def step(self, state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
        """Perform a single iteration step."""
        value, args, iteration, _, linear_solver_state = state

        residual = linear_solver_state.fn(value, args)
        jacobian_matrix = linear_solver_state.jacobian(value, args)

        # Solve for the Newton step: J * delta = -residual
        delta = linear_solver_state.linsolve(jacobian_matrix, -residual)

        new_value = value + delta

        # Check convergence
        new_converged = jax.numpy.linalg.norm(residual) < self.options.tol

        return state._replace(
            value=new_value, iteration=iteration + 1, converged=new_converged
        )

    def terminate(self, state: NewtonState[Y, Args]) -> Array:
        """Check if the solver should terminate."""
        return jnp.logical_or(
            state.converged, state.iteration >= self.options.max_iterations
        )


def find_root(
    x0: Y,
    args: Args,
    solver: _Solver[SolverState, Y, Args, SolverOptions],
) -> Any:
    """Find the root of a function using a dense Newton-Raphson solver.

    Args:
        args: Arguments required by the function whose root is to be found.

    Returns:
        The root of the function.
    """
    state = solver.init(x0, args)

    def cond_fn(state: SolverState) -> Array:
        return jnp.logical_not(solver.terminate(state))

    def body_fn(state: SolverState) -> SolverState:
        return solver.step(state)

    final_state = jax.lax.while_loop(cond_fn, body_fn, state)
    return final_state.value
