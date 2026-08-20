import jax
import jax.numpy as jnp

from soldis.linear._core import LinearSolver, LinearSolverVariant
from soldis.typing import Array, Mv


class CG(LinearSolver[Mv]):
    """Conjugate Gradient linear solver for matrix-free Jacobians.

    Inner products are written ``jnp.sum(x * y)`` rather than ``jnp.vdot(x, y)``.
    So this implementation can be used with vectors whose contracted axis is sharded
    under explicit sharding.

    We have to use "jnp.sum" instead of "jax.lax.psum" as we want to perform all reduce on
    all devices and "jax.lax.psum" needs an axis name and currently we want solver to be
    agnostic to partitioning.

    A subclass that adds fields **must** also override ``tree_flatten`` and
    ``tree_unflatten``; otherwise it inherits the ones below and silently loses
    those fields on unflatten.  Default new fields to ``None`` so that mistake
    fails loudly instead of degrading to an unpreconditioned solve.
    """

    variant = LinearSolverVariant.MATRIX_FREE

    def __init__(self, tol: float = 1e-10, maxiter: int = 100) -> None:
        self.tol = tol
        self.maxiter = maxiter

    def preconditioner(self, v: Array) -> Array:
        """Apply ``M^-1`` to ``v``.  The default is the identity -- no preconditioning."""
        return v

    def tree_flatten(self):
        return (), (self.tol, self.maxiter)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tol, maxiter = aux_data
        return cls(tol=tol, maxiter=maxiter)

    def __call__(self, A: Mv, b: Array) -> tuple[Array, tuple[Array, int]]:
        M = self.preconditioner

        x = jnp.zeros_like(b)  # derived from b, so it inherits b's sharding
        r = b - A(x)
        p = M(r)
        rsold = jnp.sum(
            r * p
        )  # replace jnp.vdot with jnp.sum for distributed sharding compatibility

        def cond_fn(state):
            _, _, _, rsold, i = state
            return jnp.logical_and(jnp.sqrt(rsold) > self.tol, i < self.maxiter)

        def body_fn(state):
            x, r, p, rsold, i = state
            Ap = A(p)
            alpha = rsold / jnp.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            z = M(r)
            rsnew = jnp.sum(
                r * z
            )  # replace jnp.vdot with jnp.sum for distributed sharding compatibility
            p = z + (rsnew / rsold) * p
            return x, r, p, rsnew, i + 1

        x, _, _, r_norm, iiter = jax.lax.while_loop(
            cond_fn, body_fn, (x, r, p, rsold, jnp.asarray(0))
        )
        return x, (r_norm, iiter)  # return residual norm and number of iterations


class JaxCG(LinearSolver[Mv]):
    """Conjugate Gradient wrapping ``jax.scipy.sparse.linalg.cg``.

    Note JAX's default has its inner products use ``jnp.vdot``, so it cannot
    operate on a vector whose contracted axis is sharded under explicit sharding.
    Prefer :class:`CG` for distributed problems.
    """

    variant = LinearSolverVariant.MATRIX_FREE

    def __init__(self, tol: float = 1e-10, maxiter: int = 100, M: Mv = None) -> None:
        self.tol = tol
        self.maxiter = maxiter
        if M is None:
            M = lambda v: v
        self.preconditioner = M

    def __call__(self, A: Mv, b: Array) -> tuple[Array, tuple[Array, int]]:
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, M=self.preconditioner, tol=self.tol, maxiter=self.maxiter
        )
        # NOTE: Currently, info is always None in JAX's CG implementation
        # we skip checking convergence for now
        return x, (
            jnp.asarray(0),
            jnp.asarray(0),
        )  # return residual norm and number of iterations
