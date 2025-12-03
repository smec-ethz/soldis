import jax

from soldis.linear._core import LinearSolver, LinearSolverVariant
from soldis.typing import Array, Mv


class GMRES(LinearSolver[Mv]):
    """GMRES linear solver for matrix-free Jacobians."""

    variant = LinearSolverVariant.MATRIX_FREE

    def __call__(self, A: Mv, b: Array) -> Array:
        x, info = jax.scipy.sparse.linalg.gmres(A, b)
        # JAX's GMRES currently returns info=None on success; skip checks for now.
        return x
