from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Concatenate, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.tree_util import register_pytree_node_class

from soldis.linear._core import DirectLinearSolver, LinearSolver, LinearSolverVariant
from soldis.typing import ArgsTuple, Fn, JacobianFunc, JacobianT, P, Y


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


def _cg_solve(
    matvec: Callable[[Array], Array], b: Array, tol: float = 1e-10, maxiter: int = 100
) -> Array:
    """Raw CG implementation using while_loop.

    Used as the inner solver for custom_linear_solve — handles only the
    forward solve, while custom_linear_solve manages differentiation.
    """
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r
    rsold = jnp.vdot(r, r)

    def cond_fn(state):
        _, _, _, rsold, i = state
        return jnp.logical_and(jnp.sqrt(rsold) > tol, i < maxiter)

    def body_fn(state):
        x, r, p, rsold, i = state
        Ap = matvec(p)
        alpha = rsold / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = jnp.vdot(r, r)
        p = r + (rsnew / rsold) * p
        return x, r, p, rsnew, i + 1

    x, _, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (x, r, p, rsold, jnp.asarray(0))
    )
    return x


def _tangent_linear_solve(matvec: Callable[[Y], Array], b: Array) -> Y:
    """Solve the tangent linear system for implicit differentiation of a root.

    Called by ``jax.lax.custom_root`` during differentiation.  When the forward
    pass finds ``x*`` such that ``f(x*, p) = 0``, the backward pass needs::

        dx*/dp = -(df/dx)^{-1}  df/dp

    ``custom_root`` supplies:

    - ``matvec = df/dx |_{x*}`` — the Jacobian as a callable (JVP), so
      ``matvec(v)`` computes ``J @ v``
    - ``b = -df/dp · dp`` — the RHS vector

    This function solves ``J @ dx = b`` for ``dx`` using CG.

    We wrap the raw CG in ``custom_linear_solve`` (rather than calling
    ``jax.scipy.sparse.linalg.cg``, which uses ``custom_linear_solve``
    internally) because nesting two ``custom_linear_solve`` calls — or nesting
    one inside ``custom_root`` �� triggers an unsupported-primitive error in JAX.
    By providing our own ``custom_linear_solve`` with a plain ``while_loop`` CG
    as the inner solver, JAX gets exactly one custom-differentiation boundary at
    the tangent level.
    """

    def solve(mv, b):
        return _cg_solve(mv, b)

    return jax.lax.custom_linear_solve(
        matvec, b, solve=solve, transpose_solve=solve, symmetric=True
    )


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

    def __init__(
        self,
        fn: Fn[Y, P],
        lin_solver: LinearSolver[JacobianT] | None = None,
        jac: JacobianFunc[Y, P, JacobianT] | None = None,
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

    def _root(self, y0: Y, *args: P.args) -> SolverState[Y, P]:
        """Raw Newton iteration without implicit differentiation."""
        state = self.init(y0, *args)

        def cond_fn(state: SolverState[Y, P]) -> Array:
            return jnp.logical_not(self.terminate(state))

        def body_fn(state: SolverState[Y, P]) -> SolverState[Y, P]:
            return self.step(state)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        return final_state

    def root(self, y0: Y, *args: P.args) -> SolverState[Y, P]:
        """Find root with implicit differentiation via jax.lax.custom_root.

        The returned state.value carries implicit gradients (via custom_root).
        Other state fields (residual, iteration, converged) are non-differentiable
        diagnostics from the raw Newton iteration.
        """
        # Run raw Newton to get the final state and the function for custom_root.
        state = self._root(y0, *args)

        def f(x: Y) -> Array:
            return self.fn(x, *args)

        def solve(f: Callable[[Y], Array], x0: Y) -> Array:
            """Solve f(x) = 0 for x, starting from x0. Used in custom_root's forward pass.

            Here, we ignore the provided x0 and instead return the final value from the raw Newton iteration.
            """
            return state.value

        def tangent_solve(g: Callable[[Y], Array], y: Array) -> Y:
            """Solve the tangent linear system for the backward pass.

            ``custom_root`` calls this during reverse-mode AD instead of
            differentiating through the Newton iterations.

            Args:
                g: The Jacobian of ``f`` at the root ``x*``, provided as a
                    callable (JVP): ``g(v)`` computes ``(df/dx)|_{x*} @ v``.
                y: The right-hand side, derived from upstream gradients via the
                    chain rule: ``y = -df/dp · dp``.

            Returns:
                ``dx`` satisfying ``(df/dx)|_{x*} @ dx = y``, which gives the
                implicit gradient ``dx*/dp = -(df/dx)^{-1} df/dp``.
            """
            return _tangent_linear_solve(g, y)

        value_implicit = jax.lax.custom_root(f, y0, solve, tangent_solve)
        return state._replace(value=value_implicit)

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
