# civ_kernel.py — VVL-AGI 1.0 极简可运行内核
import numpy as np
import cvxpy as cp


class VVLAGI:
    """
    Minimal kernel for VVL-AGI 1.0:
    - Omega-GC structural potential (very simplified)
    - VVL intensity cap (rho_max)
    - Single-step QP with box constraints
    用于 toy 示例演示整体“趋稳”行为。
    """

    def __init__(self, rho_max=0.95, omega0=0.0):
        self.rho_max = rho_max
        self.omega = omega0

    def omega_gc(self, g):
        """
        简化版结构势 Ω：
        g: 长度为 6 的失衡向量 [g_S, g_U, g_I, g_H, g_E, g_D]
        """
        g = np.asarray(g)
        base = np.sum(g)
        hd = 2.0 * g[3] * g[5]  # H-D 交互项
        logits = 2.0 * np.sum(g) - 1.0
        smooth = np.log(1.0 + np.exp(logits))
        return base + hd + smooth

    def step(self, x, u_des, g):
        """
        单步控制：
        x: 当前状态（这里不用，只是占位）
        u_des: 任务期望动作（如推荐里的“想推高情绪/点击”的原始方向）
        g: 当前失衡向量（结构上的 gap）
        返回：u_t（在 VVL + Ω 约束下的安全动作）
        """
        u_des = np.asarray(u_des)
        n = len(u_des)
        u = cp.Variable(n)

        # 简化版：期望逼近 u_des，同时惩罚 Omega 增大
        # （实际系统里会加 CBF/CLF 等）
        omega_now = self.omega_gc(g)
        omega_next = self.omega_gc(g + 0.1 * u)  # 粗略一阶近似

        cost = 0.5 * cp.sum_squares(u - u_des) \
             + 10.0 * cp.maximum(omega_next - omega_now, 0)

        constraints = [
            u >= -self.rho_max,
            u <= self.rho_max
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            return np.clip(u_des, -0.9, 0.9)

        if u.value is None:
            return np.clip(u_des, -0.9, 0.9)
        return np.array(u.value).reshape(-1)
