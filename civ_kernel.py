# civ_kernel.py — Minimal runnable kernel for VVL-AGI 1.0
import numpy as np
import cvxpy as cp


class VVLAGI:
    """
    Minimal kernel for VVL-AGI 1.0.

    This class implements a very small, self-contained version of the
    VVL-AGI control idea, intended only for toy demonstrations:

    - Omega-GC structural potential (highly simplified)
    - VVL-style intensity cap (rho_max)
    - Single-step QP with box constraints

    It is used in toy examples to show the basic "stabilising" behaviour
    of the CivKernel, not as a production implementation.
    """

    def __init__(self, rho_max: float = 0.95, omega0: float = 0.0) -> None:
        """
        Parameters
        ----------
        rho_max : float
            Maximum absolute intensity allowed on each control dimension.
            This corresponds to the VVL cap (ρ_max ≤ 0.95 in the paper).
        omega0 : float
            Initial structural potential value. In this toy version we
            only track the current value for illustration.
        """
        self.rho_max = rho_max
        self.omega = omega0

    def omega_gc(self, g):
        """
        Simplified structural potential Ω(g).

        Parameters
        ----------
        g : array-like of length 6
            Imbalance vector [g_S, g_U, g_I, g_H, g_E, g_D].

        Returns
        -------
        float
            Structural potential Ω(g). Larger values mean "further away"
            from the desired civilizational region.
        """
        g = np.asarray(g, dtype=float)

        # Base linear term over all six gaps
        base = np.sum(g)

        # Simple interaction term between H and D components
        hd = 2.0 * g[3] * g[5]

        # Smooth penalty that grows quickly when total imbalance is large
        logits = 2.0 * np.sum(g) - 1.0
        smooth = np.log(1.0 + np.exp(logits))

        return float(base + hd + smooth)

    def step(self, x, u_des, g):
        """
        One-step control with a very small QP.

        Parameters
        ----------
        x : array-like
            Current state. In this toy implementation it is unused and
            kept only for interface completeness.
        u_des : array-like
            Desired action suggested by the underlying task model
            (e.g., "push more engagement / emotion" in a recommender).
        g : array-like
            Current structural imbalance vector (same format as in
            `omega_gc`).

        Returns
        -------
        numpy.ndarray
            Controlled action u_t after applying VVL caps and Ω-based
            penalty. If the QP fails to solve, falls back to a clipped
            version of u_des.
        """
        u_des = np.asarray(u_des, dtype=float)
        n = len(u_des)
        u = cp.Variable(n)

        # Very simple objective:
        #   1) stay close to u_des
        #   2) penalize any increase in structural potential Ω
        omega_now = self.omega_gc(g)
        omega_next = self.omega_gc(g + 0.1 * u)  # crude first-order proxy

        cost = (
            0.5 * cp.sum_squares(u - u_des)
            + 10.0 * cp.maximum(omega_next - omega_now, 0)
        )

        # VVL-style intensity caps (box constraints)
        constraints = [
            u >= -self.rho_max,
            u <= self.rho_max,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            # If the solver fails, fall back to a safely clipped u_des
            return np.clip(u_des, -0.9, 0.9)

        if u.value is None:
            # Another safety fallback in case the solver returns no value
            return np.clip(u_des, -0.9, 0.9)

        return np.array(u.value, dtype=float).reshape(-1)
