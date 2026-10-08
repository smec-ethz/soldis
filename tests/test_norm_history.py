import jax
import jax.numpy as jnp
import pytest

from soldis.newton import LineSearchNewtonSolver, NewtonSolver


@pytest.mark.parametrize("solver_cls", [NewtonSolver, LineSearchNewtonSolver])
@pytest.mark.parametrize(
    "maxiter, initial, steps", [(3, 0.0, 1), (3, 2.0, 0), (0, 0.0, 0)]
)
def test_norm_history(solver_cls, maxiter, initial, steps):
    def residual(x, *, target):
        return x - target

    solver = solver_cls(residual, maxiter=maxiter)
    x0 = jnp.array([initial])
    state = jax.jit(solver.root)(x0, target=jnp.array([2.0]))

    assert state.norm_history.shape == (maxiter + 1,)
    assert int(state.iteration) == steps
    assert jnp.allclose(state.norm_history[0], abs(initial - 2.0))
    assert jnp.allclose(state.norm_history[steps], state.norm)
    assert jnp.allclose(state.norm, jnp.linalg.norm(state.residual))
    assert jnp.all(jnp.isnan(state.norm_history[steps + 1 :]))


@pytest.mark.parametrize("solver_cls", [NewtonSolver, LineSearchNewtonSolver])
def test_norm_history_implicit_derivative(solver_cls):
    solver = solver_cls(lambda x, p: x - p, maxiter=3)

    def outputs(p):
        state = solver.root(jnp.zeros(1), p)
        return state.value, state.norm_history

    (_, _), (value_dot, history_dot) = jax.jvp(
        outputs, (jnp.array([2.0]),), (jnp.ones(1),)
    )
    assert jnp.allclose(value_dot, 1.0)
    assert jnp.allclose(history_dot, 0.0)
    assert jnp.allclose(
        jax.grad(lambda p: jnp.sum(outputs(p)[0]))(jnp.array([2.0])), 1.0
    )
