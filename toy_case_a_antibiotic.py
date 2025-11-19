"""
toy_case_a_antibiotic.py

VVL-AGI 1.0 - Case A: Global Antibiotic Stewardship (toy simulation)

This script simulates the trade-off between short-term mortality reduction
and long-term antimicrobial resistance (AMR) under three policies:

  1) laissez_faire  -- maximize immediate survival (max antibiotic usage)
  2) lockdown       -- minimize usage (very conservative)
  3) vvl_agi        -- governed by a simple QP controller with an AMR barrier

It is intentionally simple and meant only as a demonstrative toy example.
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def run_simulation(mode: str = "vvl_agi", seed: int = 0):
    """Run one antibiotic policy simulation."""
    rng = np.random.default_rng(seed)
    T = 50  # 50 time steps ~ years

    # State:
    # x[0]: AMR level (0-1)
    # x[1]: healthcare quality (0-1)  [kept constant in this toy model]
    x = np.zeros((2, T + 1))
    x[:, 0] = [0.1, 0.8]  # low AMR, high quality

    u_history = []       # antibiotic usage
    mort_history = []    # mortality

    # Dynamics parameters
    alpha_kill = 0.5   # how effective antibiotics are at reducing mortality
    beta_resist = 0.05 # how much usage increases AMR
    gamma_decay = 0.01 # natural decay of AMR

    print(f"[Case A] Starting simulation: mode = {mode}")

    for t in range(T):
        # Decision variable: antibiotic usage (0-1)
        u = cp.Variable(1)

        # Predicted AMR next step (affine)
        amr_next = x[0, t] + beta_resist * u - gamma_decay

        # Simple mortality proxy at time t (lower is better)
        # mortality = base - kill * u + penalty * current AMR
        mortality = 0.2 - alpha_kill * u + 0.2 * x[0, t]

        if mode == "laissez_faire":
            objective = cp.Minimize(mortality)
            constraints = [u >= 0.0, u <= 1.0]

        elif mode == "lockdown":
            objective = cp.Minimize(mortality)
            constraints = [u >= 0.0, u <= 0.2]  # hard cap

        elif mode == "vvl_agi":
            # Structural penalty on AMR growth (Omega proxy)
            penalty_weight = 10.0 if x[0, t] > 0.3 else 1.0
            long_term_cost = penalty_weight * cp.square(amr_next)

            objective = cp.Minimize(mortality + long_term_cost)
            constraints = [
                u >= 0.0,
                u <= 0.95,        # VVL intensity cap
                amr_next <= 0.4,  # HXQ-like safety barrier
            ]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if u.value is None:
                u_val = 0.0
            else:
                u_val = float(u.value[0])
        except Exception as e:
            print(f"[Case A] QP solve failed at t={t}, mode={mode}: {e}")
            u_val = 0.0  # safe fallback

        # Environment update with small noise
        noise = rng.normal(0.0, 0.005)
        x[0, t + 1] = np.clip(
            x[0, t] + beta_resist * u_val - gamma_decay + noise, 0.0, 1.0
        )
        x[1, t + 1] = x[1, t]

        real_mortality = 0.2 - alpha_kill * u_val + 0.2 * x[0, t]
        mort_history.append(real_mortality)
        u_history.append(u_val)

    return x, np.array(mort_history), np.array(u_history)


def main():
    # Run three policies with the same seed for comparability
    x_free, m_free, u_free = run_simulation("laissez_faire", seed=0)
    x_lock, m_lock, u_lock = run_simulation("lockdown", seed=0)
    x_vvl, m_vvl, u_vvl = run_simulation("vvl_agi", seed=0)

    # Plot AMR
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_free[0, :], "r--", label="Laissez-faire (max usage)")
    plt.plot(x_lock[0, :], "g--", label="Lockdown (min usage)")
    plt.plot(x_vvl[0, :], "b-", linewidth=2, label="VVL-AGI 1.0 (governed)")
    plt.axhline(y=0.4, color="k", linestyle=":", label="Safety barrier")
    plt.title("Case A: AMR trajectories under three policies")
    plt.xlabel("Time (years)")
    plt.ylabel("AMR level")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot mortality
    plt.subplot(1, 2, 2)
    plt.plot(m_free, "r--", alpha=0.7, label="Laissez-faire")
    plt.plot(m_lock, "g--", alpha=0.7, label="Lockdown")
    plt.plot(m_vvl, "b-", linewidth=2, label="VVL-AGI 1.0")
    plt.title("Case A: Mortality trajectories")
    plt.xlabel("Time (years)")
    plt.ylabel("Mortality (proxy)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("result_case_a.png", dpi=200)
    print("[Case A] Simulation complete. Saved figure as result_case_a.png")
    plt.show()


if __name__ == "__main__":
    main()
