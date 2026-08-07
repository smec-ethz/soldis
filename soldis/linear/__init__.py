from ._core import DirectLinearSolver, LinearSolver
from .cg import CG, JaxCG
from .gmres import GMRES

try:
    from .sparse import SparseTatva
except ModuleNotFoundError:
    SparseTatva = None

__all__ = [
    "LinearSolver",
    "DirectLinearSolver",
    "CG",
    "JaxCG",
    "GMRES",
]

if SparseTatva is not None:
    __all__.append("SparseTatva")
