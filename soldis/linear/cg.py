import jax

from soldis.linear._core import LinearSolver, LinearSolverVariant
from soldis.typing import Array, Mv


class CG(LinearSolver[Mv]):
    """Conjugate Gradient linear solver for matrix-free Jacobians."""

    variant = LinearSolverVariant.MATRIX_FREE

    def __call__(self, A: Mv, b: Array) -> Array:
        x, info = jax.scipy.sparse.linalg.cg(A, b)
        # NOTE: Currently, info is always None in JAX's CG implementation
        # we skip checking convergence for now
        return x
