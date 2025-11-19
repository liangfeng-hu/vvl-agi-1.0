"""
toy_case_c_recommender.py — Tiny toy environment for Case C (recommender governance)

State (9D):
    [S, U, I, H, E, D, ViewCov, DepRisk, PolRisk]

At each step:
- The underlying model proposes a "greedy" action u_des that maximises
  engagement and emotional arousal.
- The VVLAGI kernel wraps this proposal and produces a safer control u_ctrl.
- We update the structural indicators in a crude way and record Omega and DepRisk.

Run:

    python toy_case_c_recommender.py

You should see Omega gradually decrease and DepRisk stay bounded.
"""

import numpy as np
import matplotlib.pyplot as plt

from civ_kernel import VVLAGI


def update_state(g: np.ndarray, u_ctrl: np.ndarray) -> np.ndarray:
    """
    Crude state update for the toy example.

    Parameters
    ----------
    g : np.ndarray
        6D structural indicators [S,U,I,H,E,D] in [0,1].
    u_ctrl : np.ndarray
        Governed control action.

    Returns
    -------
    np.ndarray
        Updated structural indicators g_next.
    """
    g = g.copy()
    u1, u2, u3 = u_ctrl

    # U: well-being increases with u1 but too much action increases H and D.
    g[1] = np.clip(g[1] + 0.05 * u1, 0.0, 1.0)
    g[3] = np.clip(g[3] + 0.03 * max(u1, 0.0), 0.0, 1.0)  # tail risk
    g[5] = np.clip(g[5] + 0.03 * max(u1, 0.0), 0.0, 1.0)  # drift

    # S and I: slightly improve when actions are moderate.
    g[0] = np.clip(g[0] + 0.02 * (1.0 - abs(u2)), 0.0, 1.0)
    g[2] = np.clip(g[2] + 0.02 * (1.0 - abs(u3)), 0.0, 1.0)

    # E: evidence coverage decreases if we push too aggressively.
    g[4] = np.clip(g[4] - 0.04 * (abs(u1) + abs(u2)), 0.0, 1.0)

    # Small natural recovery towards a neutral baseline.
    g = 0.98 * g + 0.02 * 0.5

    return g


def main():
    kernel = VVLAGI()
    # Initial 6D indicators [S,U,I,H,E,D]
    g = np.array([0.5, 0.5, 0.5, 0.4, 0.6, 0.4], dtype=float)

    # Underlying model's greedy action (too aggressive)
    u_des = np.array([0.9, 0.8, -0.7], dtype=float)

    omegas = []
    deprisks = []

    for t in range(200):
        omega = kernel.omega_gc(g)
        omegas.append(omega)

        deprisk = 0.5 * g[1] + 0.5 * g[3]
        deprisks.append(deprisk)

        u_ctrl = kernel.step(g, u_des, g)

        print(
            f"Step {t:3d}: Omega={omega:6.3f}, DepRisk={deprisk:5.3f}, "
            f"u_des={u_des}, u_ctrl={u_ctrl}"
        )

        g = update_state(g, u_ctrl)

    steps = np.arange(len(omegas))
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel("step")
    ax1.set_ylabel("Omega", color="tab:red")
    ax1.plot(steps, omegas, color="tab:red", label="Omega")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("DepRisk", color="tab:blue")
    ax2.plot(steps, deprisks, color="tab:blue", linestyle="--", label="DepRisk")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    plt.title("Toy Case C: Omega and DepRisk under governed control")
    plt.show()


if __name__ == "__main__":
    main()
