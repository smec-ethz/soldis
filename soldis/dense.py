from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
)

import jax
import jax.numpy as jnp
from jax import Array

Y = TypeVar("Y", bound=Array)  # could be a pytree, but likely a jax.Array
Args = TypeVar("Args")

Fn: TypeAlias = Callable[[Y, Args], Array]
Mv: TypeAlias = Callable[[Array], Array]


class _LinearOperatorProtocol(Protocol, Generic[Y, Args]):
    def linsolve(self, A: Array | Mv, b: Array) -> Array: ...
    def fn(self, y: Y, args: Args) -> Array: ...
    def linearize(self, y: Y, args: Args) -> Array | Mv: ...


class DirectDense(Generic[Y, Args]):
    """Direct dense linear solver using JAX's built-in solver."""

    linearize: Callable[[Y, Args], Array]

    def __init__(self, fn: Fn[Y, Args], jac: Callable[[Y, Args], Array]) -> None:
        self.fn = fn
        self.linearize = jac or jax.jacfwd(fn)

        # sanity checks
        assert callable(self.linearize), "Linearize must be callable."
        assert callable(self.fn), "Function must be callable."

    def linsolve(self, A: Array, b: Array) -> Array:
        return jax.numpy.linalg.solve(A, b)


class NewtonState(NamedTuple, Generic[Y, Args]):
    value: Y
    args: Args
    iteration: Array  # int
    converged: Array  # bool


class SolverOptions(NamedTuple):
    maxiter: int = 50
    tol: float = 1e-10
    verbose: bool = False


SolverOptionsT = TypeVar("SolverOptionsT", bound=SolverOptions)


class _Solver(ABC, Generic[Y, Args, SolverOptionsT]):
    fn: Fn[Y, Args]
    linear_operator: _LinearOperatorProtocol[Y, Args]
    options: SolverOptionsT

    def __init__(
        self, fn: Fn[Y, Args], linear_operator: _LinearOperatorProtocol[Y, Args]
    ) -> None:
        self.fn = fn
        self.linear_operator = linear_operator

    def find_root(
        self, y0: Y, args: Args, *, jax_loop_type: Literal["while", "scan"] = "while"
    ) -> Y:
        state = self.init(y0, args)

        def cond_fn(state: NewtonState) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
            return self.step(state)

        if jax_loop_type == "scan":

            def scan_body_fn(
                carry: NewtonState[Y, Args], _: Any
            ) -> tuple[NewtonState[Y, Args], None]:
                new_state = self.step(carry)
                return new_state, None

            max_iters = self.options.maxiter
            states, _ = jax.lax.scan(scan_body_fn, state, None, length=max_iters)

            final_state = states
        else:
            final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state.value

    @abstractmethod
    def init(self, y0: Y, args: Args, **options: Any) -> NewtonState[Y, Args]: ...

    @abstractmethod
    def step(self, state: NewtonState) -> NewtonState: ...

    @abstractmethod
    def terminate(self, state: NewtonState) -> Array: ...

    def linearize(self, y: Y, args: Args) -> Array | Mv:
        return self.linear_operator.linearize(y, args)


class NewtonSolver(_Solver[Y, Args, SolverOptions]):
    """Dense Newton-Raphson solver implementation."""

    def init(
        self,
        y0: Y,
        args: Args,
        *,
        maxiter: int = 50,
        tol: float = 1e-10,
        verbose: bool = False,
    ) -> NewtonState[Y, Args]:
        self.options = SolverOptions(maxiter=maxiter, tol=tol, verbose=verbose)

        initial_residual = self.fn(y0, args)
        initial_converged = jax.numpy.linalg.norm(initial_residual) < self.options.tol

        return NewtonState(
            value=y0,
            args=args,
            iteration=jnp.asarray(0),
            converged=initial_converged,
        )

    def step(self, state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
        """Perform a single iteration step."""
        value, args, iteration, _ = state

        residual = self.fn(value, args)
        jacobian_matrix = self.linearize(value, args)

        # Solve for the Newton step: J * delta = -residual
        delta = self.linear_operator.linsolve(jacobian_matrix, -residual)

        new_value = value + delta

        # Check convergence
        new_converged = jax.numpy.linalg.norm(residual) < self.options.tol

        return state._replace(
            value=new_value, iteration=iteration + 1, converged=new_converged
        )

    def terminate(self, state: NewtonState[Y, Args]) -> Array:
        """Check if the solver should terminate."""
        return jnp.logical_or(state.converged, state.iteration >= self.options.maxiter)


def find_root(
    x0: Y,
    args: Args,
    solver: _Solver[Y, Args, SolverOptionsT],
) -> Any:
    """Find the root of a function using a dense Newton-Raphson solver.

    Args:
        args: Arguments required by the function whose root is to be found.

    Returns:
        The root of the function.
    """
    state = solver.init(x0, args)

    def cond_fn(state: NewtonState) -> Array:
        return jnp.logical_not(solver.terminate(state))

    def body_fn(state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
        return solver.step(state)

    final_state = jax.lax.while_loop(cond_fn, body_fn, state)
    return final_state.value
