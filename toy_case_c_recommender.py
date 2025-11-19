"""
toy_case_c_recommender.py

VVL-AGI 1.0 - Case C: Platform-scale Recommender Governance (toy simulation)

We simulate a simple two-dimensional user state:

  x[0]: depression risk (0-1)
  x[1]: polarization index (0-1)

Actions:
  u[0] = engagement boost (dopamine-like stimulation)
  u[1] = diversity / safety injection (cooling + multi-perspective content)

The QP controller trades off engagement vs. structural risk, subject to
barrier-like constraints on depression and polarization.
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def smooth(y, box_pts: int = 10):
    """Moving average smoothing for plotting."""
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def run_recommender_sim(seed: int = 0):
    rng = np.random.default_rng(seed)
    T = 200

    # x[0]: depression risk, x[1]: polarization
    x = np.zeros((2, T + 1))
    x[:, 0] = [0.2, 0.2]  # healthy start

    metrics = {
        "engagement": [],
        "view_cov": [],
        "dep_risk": [],
        "pol_risk": [],
    }

    print("[Case C] Starting recommender simulation...")

    for t in range(T):
        # Natural drift: without governance, users slowly become more at risk
        user_drift = np.array([0.01, 0.01])

        # Decision variables:
        # u[0]: engagement boost
        # u[1]: diversification / safety injection
        u = cp.Variable(2)

        dep_next = x[0, t] + 0.10 * u[0] - 0.08 * u[1] + user_drift[0]
        pol_next = x[1, t] + 0.15 * u[0] - 0.10 * u[1] + user_drift[1]

        # Engagement proxy
        engagement_gain = u[0]

        # Structural risk (depression + polarization)
        structural_risk = cp.square(dep_next) + cp.square(pol_next)

        # Objective: maximize engagement - lambda * risk
        # => minimize -engagement_gain + lambda * structural_risk
        cost = -engagement_gain + 10.0 * structural_risk

        constraints = [
            u >= 0.0,
            u <= 1.0,
            dep_next <= 0.6,  # barrier on depression risk
            pol_next <= 0.7,  # barrier on polarization
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if u.value is None:
                action = np.array([0.0, 1.0])
            else:
                action = np.array(u.value).reshape(-1)
        except Exception as e:
            print(f"[Case C] QP failed at t={t}: {e}")
            action = np.array([0.0, 1.0])  # emergency: full safety

        # Environment update with noise
        noise = rng.normal(0.0, 0.01, 2)
        x[0, t + 1] = np.clip(
            x[0, t] + 0.10 * action[0] - 0.08 * action[1] + user_drift[0] + noise[0],
            0.0,
            1.0,
        )
        x[1, t + 1] = np.clip(
            x[1, t] + 0.15 * action[0] - 0.10 * action[1] + user_drift[1] + noise[1],
            0.0,
            1.0,
        )

        # Metrics for plotting
        eng = 0.8 * action[0] + 0.2 * rng.random()
        view_cov = 0.1 + 0.8 * action[1]

        metrics["engagement"].append(eng)
        metrics["view_cov"].append(view_cov)
        metrics["dep_risk"].append(x[0, t + 1])
        metrics["pol_risk"].append(x[1, t + 1])

    return metrics


def main():
    data = run_recommender_sim(seed=0)
    t = np.arange(len(data["engagement"]))

    plt.figure(figsize=(10, 6))

    plt.plot(t, smooth(data["engagement"]), "r-", linewidth=2, label="Engagement (trend)")
    plt.plot(t, data["dep_risk"], "b-", linewidth=2, label="Depression risk")
    plt.plot(t, data["view_cov"], "g-", linewidth=2, label="Viewpoint coverage")
    plt.axhline(y=0.6, color="k", linestyle=":", label="Depression barrier")

    plt.title("Case C: Recommender Governance under VVL-AGI 1.0 (toy model)")
    plt.xlabel("Time steps")
    plt.ylabel("Normalized metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result_case_c.png", dpi=200)
    print("[Case C] Simulation complete. Saved figure as result_case_c.png")
    plt.show()


if __name__ == "__main__":
    main()
