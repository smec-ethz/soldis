from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array
from jax.experimental.sparse.linalg import spsolve

from soldis.linear._core import LinearSolver, LinearSolverVariant

if TYPE_CHECKING:
    from tatva.sparse import ColoredMatrix

_TATVA_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from tatva.sparse import ColoredMatrix
except ModuleNotFoundError as exc:
    _TATVA_IMPORT_ERROR = exc


class SparseTatva(LinearSolver[ColoredMatrix]):
    variant = LinearSolverVariant.DIRECT

    def __init__(self) -> None:
        if _TATVA_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "SparseTatva requires the optional dependency 'tatva'. "
                "Install it to use soldis.linear.SparseTatva."
            ) from _TATVA_IMPORT_ERROR

    def __call__(self, A: ColoredMatrix, b: Array) -> Array:
        return spsolve(A.data, A.indices, A.indptr, b)
