import jax
import jax.numpy as jnp

from soldis.linear._core import LinearSolver, LinearSolverVariant
from soldis.typing import Array, Mv


class GMRES(LinearSolver[Mv]):
    """GMRES linear solver for matrix-free Jacobians."""

    variant = LinearSolverVariant.MATRIX_FREE

    def __call__(self, A: Mv, b: Array) -> Array:
        x, info = jax.scipy.sparse.linalg.gmres(A, b)
        # JAX's GMRES currently returns info=None on success; skip checks for now.
        return x


class CustomGMRES(LinearSolver[Mv]):
    """Simple GMRES linear solver for matrix-free Jacobians."""

    variant = LinearSolverVariant.MATRIX_FREE

    def __call__(
        self, A: Mv, b: Array, *, tol: float = 1e-6, maxiter: int | None = None
    ) -> Array:
        n = b.shape[0]
        m = n if maxiter is None else maxiter

        beta = jnp.linalg.norm(b)
        V = jnp.zeros((m + 1, n), dtype=b.dtype)
        V = V.at[0].set(jnp.where(beta > 0, b / beta, b))
        H = jnp.zeros((m + 1, m), dtype=b.dtype)
        cs = jnp.zeros((m,), dtype=b.dtype)
        sn = jnp.zeros((m,), dtype=b.dtype)
        g = jnp.zeros((m + 1,), dtype=b.dtype)
        g = g.at[0].set(beta)

        def arnoldi_step(k, state):
            V, H, cs, sn, g, done = state

            w = A(V[k])

            def orth_body(i, w_):
                h_ik = jnp.vdot(V[i], w_)
                w_ = w_ - h_ik * V[i]
                H_updated = H.at[i, k].set(h_ik)
                return w_, H_updated

            w, H = jax.lax.fori_loop(
                0,
                k + 1,
                lambda i, carry: orth_body(i, carry[0]),
                (w, H),
            )

            h_next = jnp.linalg.norm(w)
            H = H.at[k + 1, k].set(h_next)
            V = V.at[k + 1].set(jnp.where(h_next > 0, w / h_next, V[k + 1]))

            def apply_prev_rotations(i, H_col):
                temp = cs[i] * H_col[i] + sn[i] * H_col[i + 1]
                H_col_next = -sn[i] * H_col[i] + cs[i] * H_col[i + 1]
                H_col = H_col.at[i].set(temp)
                H_col = H_col.at[i + 1].set(H_col_next)
                return H_col

            H_col = H[:, k]
            H_col = jax.lax.fori_loop(0, k, apply_prev_rotations, H_col)

            denom = jnp.sqrt(H_col[k] ** 2 + H_col[k + 1] ** 2)
            cs_k = jnp.where(
                denom == 0, jnp.asarray(1.0, dtype=b.dtype), H_col[k] / denom
            )
            sn_k = jnp.where(
                denom == 0, jnp.asarray(0.0, dtype=b.dtype), H_col[k + 1] / denom
            )

            H_col = H_col.at[k].set(cs_k * H_col[k] + sn_k * H_col[k + 1])
            H_col = H_col.at[k + 1].set(0.0)
            H = H.at[:, k].set(H_col)

            cs = cs.at[k].set(cs_k)
            sn = sn.at[k].set(sn_k)

            g_k = cs_k * g[k]
            g_k1 = -sn_k * g[k]
            g = g.at[k].set(g_k)
            g = g.at[k + 1].set(g_k1)

            resid = jnp.abs(g_k1)
            done = jnp.logical_or(done, resid <= tol)

            return V, H, cs, sn, g, done

        def cond_fn(carry):
            k, _, _, _, _, done = carry
            return jnp.logical_and(k < m, jnp.logical_not(done))

        def body_fn(carry):
            k, V, H, cs, sn, g, done = carry
            V, H, cs, sn, g, done = arnoldi_step(k, (V, H, cs, sn, g, done))
            return k + 1, V, H, cs, sn, g, done

        k0 = jnp.asarray(0)
        k_final, V, H, cs, sn, g, _ = jax.lax.while_loop(
            cond_fn, body_fn, (k0, V, H, cs, sn, g, jnp.asarray(False))
        )

        m_used = k_final

        def back_sub(i, y):
            idx = m_used - 1 - i
            rhs = g[idx] - jnp.dot(H[idx, idx + 1 : m_used], y[idx + 1 :])
            y = y.at[idx].set(rhs / H[idx, idx])
            return y

        y0 = jnp.zeros((m,), dtype=b.dtype)
        y = jax.lax.fori_loop(0, m, back_sub, y0)[:m_used]

        x = jnp.tensordot(y, V[:m_used], axes=1)
        return x
