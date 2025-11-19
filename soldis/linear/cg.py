import jax

from soldis.typing import Array, Mv


def cg(A: Mv, b: Array) -> Array:
    """Conjugate Gradient linear solver for matrix-free Jacobians."""
    x, info = jax.scipy.sparse.linalg.cg(A, b)
    if info != 0:
        raise RuntimeError(f"Conjugate Gradient did not converge: info={info}")
    return x
