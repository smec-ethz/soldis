"""Test that solver.root uses implicit differentiation (custom_root)
and produces correct gradients without unrolling Newton iterations."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions


@pytest.fixture
def linear_system_solver():
    """A Newton solver for a simple linear system: A @ x - b = 0.

    A is a fixed 3x3 SPD matrix. The parameter `p` scales the RHS:
        f(x, p) = A @ x - p * b0

    Root:      x*(p) = p * A^{-1} b0
    dx*/dp:    A^{-1} b0
    """
    A = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b0 = jnp.array([1.0, 2.0, 3.0])

    def residual(x, p):
        return A @ x - p * b0

    solver = NewtonSolver(
        residual,
        lin_solver=CG(),
        options=NewtonSolverOptions(tol=1e-12, maxiter=50),
    )
    return solver, A, b0


def test_implicit_root_correctness(linear_system_solver):
    """root() returns the correct solution."""
    solver, A, b0 = linear_system_solver
    p = jnp.array(2.0)
    x0 = jnp.zeros(3)

    state = solver.root(x0, p)
    x_expected = jnp.linalg.solve(A, p * b0)

    assert jnp.allclose(state.value, x_expected, atol=1e-10)


def test_implicit_root_gradient(linear_system_solver):
    """jax.grad through root() gives the correct implicit gradient."""
    solver, A, b0 = linear_system_solver

    def loss(p):
        x0 = jnp.zeros(3)
        state = solver.root(x0, p)
        return jnp.sum(state.value**2)

    p = jnp.array(2.0)

    # Compute gradient via implicit diff
    grad_implicit = jax.grad(loss)(p)

    # Analytical gradient:
    #   x*(p) = p * A^{-1} b0
    #   loss  = ||x*(p)||^2
    #   dloss/dp = 2 * x*(p)^T @ dx*/dp = 2 * x*(p)^T @ A^{-1} b0
    x_star = jnp.linalg.solve(A, p * b0)
    Ainv_b0 = jnp.linalg.solve(A, b0)
    grad_analytical = 2.0 * jnp.dot(x_star, Ainv_b0)

    assert jnp.allclose(grad_implicit, grad_analytical, atol=1e-8), (
        f"implicit={grad_implicit}, analytical={grad_analytical}"
    )


def test_implicit_root_gradient_vs_finite_diff(linear_system_solver):
    """Implicit gradient matches finite differences."""
    solver, A, b0 = linear_system_solver

    def loss(p):
        x0 = jnp.zeros(3)
        state = solver.root(x0, p)
        return jnp.sum(state.value**2)

    p = jnp.array(2.0)
    grad_implicit = jax.grad(loss)(p)

    # Finite differences
    eps = 1e-6
    grad_fd = (loss(p + eps) - loss(p - eps)) / (2 * eps)

    assert jnp.allclose(grad_implicit, grad_fd, atol=1e-5), (
        f"implicit={grad_implicit}, fd={grad_fd}"
    )


def test_nonlinear_implicit_gradient():
    """Implicit gradient works for a nonlinear problem: x^3 - p = 0."""

    def residual(x, p):
        return x**3 - p

    solver = NewtonSolver(
        residual,
        lin_solver=CG(),
        options=NewtonSolverOptions(tol=1e-12, maxiter=50),
    )

    def loss(p):
        x0 = jnp.array(1.0)  # initial guess
        state = solver.root(x0, p)
        return state.value**2

    p = jnp.array(8.0)
    grad_implicit = jax.grad(loss)(p)

    # x*(p) = p^(1/3), loss = p^(2/3)
    # dloss/dp = (2/3) * p^(-1/3)
    grad_analytical = (2.0 / 3.0) * p ** (-1.0 / 3.0)

    assert jnp.allclose(grad_implicit, grad_analytical, atol=1e-6), (
        f"implicit={grad_implicit}, analytical={grad_analytical}"
    )


def test_linearize_jacrev_multiscale_pattern():
    """jax.linearize(jax.jacrev(energy)) works when energy calls solver.root() per point.

    This is the FE² (multiscale) pattern: a macro energy function sums
    constitutive responses from per-quadrature-point RVE Newton solves.
    The tangent stiffness is assembled by linearizing the gradient, which
    requires second-order AD (JVP of VJP) through custom_root.

    Previously failed because the forward Newton solve ran outside custom_root's
    solve() closure, so JAX's linearize transformation flowed into the CG solver
    and passed Zero tangent sentinels as the RHS.
    """
    # RVE: nonlinear scalar spring — find x such that x^3 + k*x = eps
    k = 2.0

    def rve_residual(x, eps):
        return x**3 + k * x - eps

    rve_solver = NewtonSolver(
        rve_residual,
        lin_solver=CG(),
        options=NewtonSolverOptions(tol=1e-12, maxiter=50),
    )

    def constitutive_update(eps):
        """Solve the RVE and return the homogenized stress."""
        x0 = jnp.zeros_like(eps)
        return rve_solver.root(x0, eps).value

    @jax.jit
    def total_energy(u):
        """Macro energy: sum 0.5 * sigma(u_i) * u_i over N quadrature points."""
        sigma = jax.vmap(constitutive_update)(u)
        return 0.5 * jnp.dot(sigma, u)

    u = jnp.array([0.1, 0.2, 0.3, 0.4])
    N = u.shape[0]

    grad_fn = jax.jacrev(total_energy)

    # Linearize the gradient to get the tangent stiffness matrix
    _, fn_linear = jax.linearize(grad_fn, u)
    K = jax.vmap(fn_linear)(jnp.eye(N))

    # Reference: nested jax.grad (exact second derivative per point)
    # Points are decoupled, so K is diagonal
    K_diag_ref = jax.vmap(
        jax.grad(jax.grad(lambda eps: 0.5 * constitutive_update(eps) * eps))
    )(u)

    assert jnp.allclose(jnp.diag(K), K_diag_ref, atol=1e-8), (
        f"diagonal mismatch: K_diag={jnp.diag(K)}, ref={K_diag_ref}"
    )
    assert jnp.allclose(K, jnp.diag(K_diag_ref), atol=1e-8), (
        f"off-diagonal entries should be zero: max={jnp.abs(K - jnp.diag(jnp.diag(K))).max()}"
    )
