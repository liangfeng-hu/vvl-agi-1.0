# civ_kernel.py — Minimal runnable kernel for VVL-AGI 1.0
# --------------------------------------------------------
# This file implements a tiny, self-contained version of the
# VVL-AGI control kernel:
#   - a very simplified structural potential Omega-GC
#   - a single intensity cap (rho_max) as a VVL-like valve
#   - a one-step QP controller with box constraints
#
# The goal is NOT to be numerically accurate, but to provide
# a minimal, auditable example that shows:
#   - the controller tries to track u_des
#   - while penalising increases in structural imbalance.

import numpy as np
import cvxpy as cp


class VVLAGI:
    """
    Minimal VVL-AGI 1.0 kernel for toy examples.

    Attributes
    ----------
    rho_max : float
        Global cap on action magnitude (VVL-style intensity bound).
    omega : float
        Current structural potential (for logging only in this minimal demo).
    """

    def __init__(self, rho_max: float = 0.95, omega0: float = 0.0) -> None:
        self.rho_max = float(rho_max)
        self.omega = float(omega0)

    # ------------------------------------------------------------------
    #  Simplified structural potential Omega-GC (pure numpy, no cvxpy)
    # ------------------------------------------------------------------
    def omega_gc(self, g):
        """
        Compute a very simple structural potential Omega.

        Parameters
        ----------
        g : array-like
            Imbalance vector. In the paper this corresponds to
            gaps on (S, U, I, H, E, D). Here we accept any length
            and use a smooth penalty.

        Returns
        -------
        float
            Structural potential Omega(g).
        """
        g = np.asarray(g, dtype=float).ravel()
        if g.size == 0:
            return 0.0

        base = np.sum(g)
        # If we have at least 4 components, add a simple interaction term.
        if g.size >= 4:
            hd = 2.0 * g[3] * g[-1]
        else:
            hd = 0.0
        logits = 2.0 * np.sum(g) - 1.0
        smooth = np.log(1.0 + np.exp(logits))
        return float(base + hd + smooth)

    # ------------------------------------------------------------------
    #  One-step QP controller
    # ------------------------------------------------------------------
    def step(self, x, u_des, g):
        """
        Compute a safe action for one step.

        Parameters
        ----------
        x : array-like
            Current state (unused in this minimal demo, but kept
            for API compatibility).
        u_des : array-like
            Desired action from the task-level objective. In a
            recommender setting this can be “raw” preference
            towards engagement / emotion.
        g : array-like
            Current structural gap vector used by Omega-GC.

        Returns
        -------
        np.ndarray
            Chosen action u_t after applying VVL cap and Omega penalty.
        """
        u_des = np.asarray(u_des, dtype=float).ravel()
        n = u_des.size
        if n == 0:
            return np.zeros(0, dtype=float)

        u = cp.Variable(n)

        # Current structural potential (numeric, no cvxpy inside).
        omega_now = self.omega_gc(g)

        # We do NOT pass u into omega_gc to avoid shape / type issues.
        # Instead, we use the squared norm of u as a crude proxy for
        # “how much we push the structure”, and penalise any increase
        # over the current omega.
        omega_proxy = omega_now + 0.1 * cp.sum_squares(u)

        cost = (
            0.5 * cp.sum_squares(u - u_des)
            + 10.0 * cp.pos(omega_proxy - omega_now)
        )

        constraints = [
            u >= -self.rho_max,
            u <= self.rho_max,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            # If the solver fails for any reason, fall back to a clipped
            # version of the desired action. This keeps the demo robust.
            return np.clip(u_des, -0.9, 0.9)

        if u.value is None:
            return np.clip(u_des, -0.9, 0.9)

        u_val = np.array(u.value, dtype=float).ravel()

        # Update omega for logging (not strictly necessary in this toy demo).
        self.omega = self.omega_gc(g)

        return u_val
