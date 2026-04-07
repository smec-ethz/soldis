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
