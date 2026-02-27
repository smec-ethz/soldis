from ._core import DirectLinearSolver, LinearSolver
from .cg import CG
from .gmres import GMRES

try:
    from .sparse import SparseTatva
except ModuleNotFoundError:
    SparseTatva = None

__all__ = [
    "LinearSolver",
    "DirectLinearSolver",
    "CG",
    "GMRES",
]

if SparseTatva is not None:
    __all__.append("SparseTatva")
