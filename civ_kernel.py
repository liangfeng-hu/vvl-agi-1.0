# civ_kernel.py — minimal runnable kernel for VVL-AGI 1.0
import numpy as np
import cvxpy as cp


class VVLAGI:
    """
    Minimal kernel for VVL-AGI 1.0.

    It implements three ideas in a very compact form:
      - Omega-GC structural potential (highly simplified)
      - VVL-style intensity cap (rho_max)
      - One-step QP with box constraints

    This is only a toy kernel used by toy_case_c_recommender.py
    to show “stabilising behaviour” of the framework.
    """

    def __init__(self, rho_max: float = 0.95, omega0: float = 0.0) -> None:
        self.rho_max = float(rho_max)
        self.omega = float(omega0)

    def omega_gc(self, g):
        """
        Simplified structural potential Ω.

        Parameters
        ----------
        g : array-like, shape (6,)
            Gap vector [g_S, g_U, g_I, g_H, g_E, g_D].

        Returns
        -------
        float
            Scalar structural potential.
        """
        g = np.asarray(g, dtype=float).reshape(-1)
        base = np.sum(g)

        # simple H-D interaction term if we have at least 6 dims
        hd = 0.0
        if g.size >= 6:
            hd = 2.0 * g[3] * g[5]

        logits = 2.0 * np.sum(g) - 1.0
        smooth = float(np.log(1.0 + np.exp(logits)))
        return float(base + hd + smooth)

    def step(self, x, u_des, g):
        """
        One-step control update.

        Parameters
        ----------
        x : array-like
            Current state (unused in this toy kernel, kept for API compatibility).
        u_des : array-like
            Task-desired action (e.g. “push engagement/emotion up”).
        g : array-like
            Current structural gap vector.

        Returns
        -------
        np.ndarray
            Safe action u_t after applying VVL cap and Ω penalty.
        """
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = len(u_des)
        u = cp.Variable(n)

        # current structural potential (NumPy only)
        omega_now = self.omega_gc(g)

        # proxy for potential increase: proportional to the L1 norm of u
        # (do NOT call omega_gc with cvxpy variables)
        omega_increase = 0.1 * cp.norm1(u)

        cost = 0.5 * cp.sum_squares(u - u_des) \
             + 10.0 * cp.maximum(omega_increase, 0.0)

        constraints = [
            u >= -self.rho_max,
            u <= self.rho_max,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            # fall back to a clipped version of u_des
            return np.clip(u_des, -0.9, 0.9)

        if u.value is None:
            return np.clip(u_des, -0.9, 0.9)

        return np.array(u.value, dtype=float).reshape(-1)
