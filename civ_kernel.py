# civ_kernel.py — Minimal runnable kernel for VVL-AGI 1.0
import numpy as np
import cvxpy as cp


class VVLAGI:
    """
    Minimal kernel for VVL-AGI 1.0 (toy version):
    - Omega-GC structural potential (highly simplified)
    - VVL intensity cap (rho_max)
    - Single-step QP with box constraints

    This is only meant for toy demos (e.g. Case C recommender),
    not for production use.
    """

    def __init__(self, rho_max: float = 0.95) -> None:
        # maximal allowed intensity (VVL cap)
        self.rho_max = float(rho_max)

    def omega_gc(self, g) -> float:
        """
        Simplified structural potential Ω.

        Parameters
        ----------
        g : array-like of length 6
            [g_S, g_U, g_I, g_H, g_E, g_D] = gaps on 6 dimensions.

        Returns
        -------
        float
            Scalar structural potential.
        """
        g = np.asarray(g, dtype=float).reshape(-1)
        base = float(np.sum(g))
        hd = 0.0
        if g.size >= 6:
            hd = 2.0 * g[3] * g[5]  # interaction between H and D
        logits = 2.0 * np.sum(g) - 1.0
        smooth = float(np.log1p(np.exp(logits)))
        return base + hd + smooth

    def step(self, x, u_des, g):
        """
        One-step controlled action.

        Parameters
        ----------
        x : np.ndarray
            Current state (unused in this toy version, kept for API symmetry).
        u_des : array-like
            Desired action from the underlying model (task objective only).
        g : array-like
            Current structural gap vector used to compute Ω.

        Returns
        -------
        np.ndarray
            Safe action u_t after applying VVL cap and Ω-penalty.
        """
        u_des = np.asarray(u_des, dtype=float).reshape(-1)
        n = u_des.size
        u = cp.Variable(n)

        # current structural potential (scalar constant)
        omega_now = self.omega_gc(g)

        # tracking term: keep close to u_des
        tracking = 0.5 * cp.sum_squares(u - u_des)

        # structural penalty: when Ω is large, penalise large |u|
        structural_penalty = 0.1 * omega_now * cp.sum_squares(u)

        cost = tracking + structural_penalty

        constraints = [
            u >= -self.rho_max,
            u <= self.rho_max,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            # fall back to clipped u_des if solver fails
            return np.clip(u_des, -self.rho_max, self.rho_max)

        if u.value is None:
            return np.clip(u_des, -self.rho_max, self.rho_max)

        return np.array(u.value, dtype=float).reshape(-1)
