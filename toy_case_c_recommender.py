# toy_case_c_recommender.py — Case C 演示脚本
import numpy as np
from civ_kernel import VVLAGI


def simulate_case_c(T=100):
    """
    简化版 Case C：
    - g: [g_S, g_U, g_I, g_H, g_E, g_D] 六维失衡
    - u_des: 想要“推高点击+情绪”的原始动作
    - kernel.step: 在 VVL+Omega 约束下给出安全 u_t
    """
    kernel = VVLAGI()
    # 初始失衡：H, D 比较大，表示尾部风险高、漂移大
    g = np.array([0.2, 0.3, 0.2, 0.8, 0.4, 0.7])
    x = np.zeros(3)  # 状态占位
    u_des = np.array([0.8, 0.9, -0.5])  # 想推高 engagement、情绪，压制多样性

    omega_list = []
    dep_list = []
    viewcov_list = []

    for t in range(T):
        u = kernel.step(x, u_des, g)

        # 简单的“环境响应”：u 越大，DepRisk 越高，ViewCov 越低
        dep_risk = 0.5 + 0.3 * u[0]  # 假设与第一个维度相关
        view_cov = 0.5 - 0.2 * u[1]  # 与第二个维度反向
        dep_risk = float(np.clip(dep_risk, 0, 1))
        view_cov = float(np.clip(view_cov, 0, 1))

        # 用 DepRisk / ViewCov 反向更新结构失衡 g（只是 toy 逻辑）
        g[3] = dep_risk        # g_H
        g[5] = 1 - view_cov    # g_D

        omega = kernel.omega_gc(g)
        omega_list.append(omega)
        dep_list.append(dep_risk)
        viewcov_list.append(view_cov)

        print(f"t={t:03d}  u={u}  Ω={omega:.3f}  DepRisk={dep_risk:.3f}  ViewCov={view_cov:.3f}")

    return np.array(omega_list), np.array(dep_list), np.array(viewcov_list)


if __name__ == "__main__":
    simulate_case_c(T=50)
