from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    NamedTuple,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

import jax
import jax.numpy as jnp
from jax import Array

Y = TypeVar("Y", bound=Array)  # could be a pytree, but likely a jax.Array
Args = TypeVar("Args")

Fn: TypeAlias = Callable[[Y, Args], Array]
Mv: TypeAlias = Callable[[Array], Array]  # Matrix-vector product function
Jacobian: TypeAlias = Array | Mv
JacobianT = TypeVar("JacobianT", bound=Jacobian)
LinearSolver: TypeAlias = Callable[[JacobianT, Array], Array]


def cg(A: Mv, b: Array) -> Array:
    """Conjugate Gradient linear solver for matrix-free Jacobians."""
    x, info = jax.scipy.sparse.linalg.cg(A, b)
    if info != 0:
        raise RuntimeError(f"Conjugate Gradient did not converge: info={info}")
    return x


class LinearOperator(NamedTuple, Generic[Y, Args, JacobianT]):
    solve_fn: LinearSolver[JacobianT]
    fn: Fn[Y, Args]
    jac: Callable[[Y, Args], JacobianT]

    @classmethod
    def from_fn(cls, solve_fn: LinearSolver[JacobianT], fn: Fn[Y, Args]) -> Self:
        jac = jax.jacfwd(fn)
        return cls(solve_fn, fn, jac)


def _compute_increment(
    self, lin_op: LinearOperator, args: tuple[Y, Args], b: Array
) -> Array:
    A = lin_op.jac(*args)
    return lin_op.solve_fn(A, b)


class NewtonState(NamedTuple, Generic[Y, Args]):
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


@dataclass(frozen=True)
class NewtonSolverOptions(SolverOptions):
    norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm


class _Solver(ABC, Generic[SolverOptionsT, Y, Args, JacobianT]):
    options: SolverOptionsT
    fn: Fn[Y, Args]

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

    def root(self, y0: Y, args: Args) -> NewtonState[Y, Args]:
        state = self.init(y0, args)

        def cond_fn(state: NewtonState) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
            return self.step(state)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state

    @abstractmethod
    def init(self, y0: Y, args: Args) -> NewtonState[Y, Args]: ...

    @abstractmethod
    def step(self, state: NewtonState) -> NewtonState: ...

    @abstractmethod
    def terminate(self, state: NewtonState) -> Array: ...


class NewtonSolver(_Solver[NewtonSolverOptions, Y, Args, JacobianT]):
    """Dense Newton-Raphson solver implementation."""

    def __init__(
        self,
        fn: Fn[Y, Args],
        jac: Callable[[Y, Args], JacobianT] | None = None,
        lin_solve: LinearSolver[JacobianT] | None = None,
        *,
        maxiter: int = 50,
        tol: float = 1e-10,
        verbose: bool = False,
        norm_fn: Callable[[Array], Array] = jax.numpy.linalg.norm,
    ) -> None:
        super().__init__(fn, jac, lin_solve)
        self.options = NewtonSolverOptions(
            maxiter=maxiter, tol=tol, verbose=verbose, norm_fn=norm_fn
        )

    def init(
        self,
        y0: Y,
        args: Args,
    ) -> NewtonState[Y, Args]:
        initial_residual = self.fn(y0, args)
        initial_converged = jax.numpy.linalg.norm(initial_residual) < self.options.tol

        return NewtonState(
            value=y0,
            args=args,
            residual=initial_residual,
            iteration=jnp.asarray(0),
            converged=initial_converged,
        )

    def step(self, state: NewtonState[Y, Args]) -> NewtonState[Y, Args]:
        """Perform a single iteration step."""
        delta = _compute_increment(
            self, self.lin_op, (state.value, state.args), -state.residual
        )
        new_value = cast(Y, state.value + delta)

        # Check convergence
        new_residual = self.fn(new_value, state.args)
        new_converged = jax.numpy.linalg.norm(new_residual) < self.options.tol

        return state._replace(
            value=new_value,
            residual=new_residual,
            iteration=state.iteration + 1,
            converged=new_converged,
        )

    def terminate(self, state: NewtonState[Y, Args]) -> Array:
        """Check if the solver should terminate."""
        return jnp.logical_or(state.converged, state.iteration >= self.options.maxiter)
