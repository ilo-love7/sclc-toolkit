# -*- coding: utf-8 -*-
"""SCLC 物理模型与拟合器（供 pinn_sclc.py 与单元测试共用）。

- sclc_curve: 欧姆区 → 平滑过渡 → SCLC/Mark-Helfrich 区的合成 J-V 模型
- fit_physics: 物理约束参数辨识（联合优化 μ, n₀, V_TFL, m）
- fit_lsq_window / fit_lsq_all: 两种传统 V² 拟合（作对照基线）
"""
import numpy as np
import torch

# ------------------------- 物理常数与器件参数 -------------------------
Q = 1.602e-19        # 元电荷 C
EPS0 = 8.854e-12     # 真空介电常数 F/m
L = 20e-9            # SnO₂ 厚度 20 nm（实验方案给定）
EPS_R = 3.0          # 相对介电常数（实验方案给定）

MU_TRUE = 2e-7       # 真值电子迁移率 m²/V·s (= 2e-3 cm²/V·s)
N0_TRUE = 2.5e23     # 真值热生背景载流子浓度 m⁻³
V_TFL_TRUE = Q * N0_TRUE * L**2 / (2 * EPS_R * EPS0)  # Lampert 陷阱填充极限
V_MAX = 1.3


def sclc_curve(v, mu, n0, v_tfl, m=2.0, width=0.04):
    """合成暗态 J-V 模型：欧姆区 → 平滑过渡 → SCLC/Mark-Helfrich 区。

    J_ohm  = q·μ·n₀·V/L                      （欧姆区，斜率1）
    J_sclc = (9/8)·εε₀·μ·V²/L³·(V/1V)^(m-2)  （m=2 退化为 Mott-Gurney；
                                              m>2 对应指数陷阱 Mark-Helfrich）
    J = (1−g)·J_ohm + g·J_sclc，g = sigmoid((V−V_TFL)/width)
    """
    v = np.asarray(v, dtype=float)
    j_ohm = Q * mu * n0 * v / L
    j_sclc = (9 / 8) * EPS_R * EPS0 * mu * v**2 / L**3 * np.maximum(v, 1e-6)**(m - 2)
    g = 1 / (1 + np.exp(-(v - v_tfl) / width))
    return (1 - g) * j_ohm + g * j_sclc


# %%

# ------------------------- 2. 物理约束拟合器（PyTorch） -------------------------
def fit_physics(v, j, n_iter=600):
    """联合优化 (μ, n₀, V_TFL, m)，全部数据点都参与，无需选窗口。

    参数化说明：
      μ, n₀, V_TFL 在对数空间搜索（跨数量级，线性空间难优化）
      m = 1 + 3·sigmoid(m_raw) ∈ (1, 4)：m→1 表示欧姆主导，
          m=2 纯 Mott-Gurney，m>2 陷阱填充。
    返回 (mu, n0, v_tfl, m)。
    """
    vt = torch.tensor(v, dtype=torch.float64)
    jt = torch.tensor(j, dtype=torch.float64)
    init = torch.tensor([np.log(1e-8), np.log(1e22), np.log(0.3), 0.0],
                        dtype=torch.float64)
    params = torch.nn.Parameter(init)
    opt = torch.optim.LBFGS([params], lr=0.3, max_iter=n_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        mu = torch.exp(params[0])
        n0 = torch.exp(params[1])
        v_tfl = torch.exp(params[2])
        m = 1 + 3 * torch.sigmoid(params[3])
        j_ohm = Q * mu * n0 * vt / L
        j_sclc = (9 / 8) * EPS_R * EPS0 * mu * vt**2 / L**3 * vt**(m - 2)
        g = torch.sigmoid((vt - v_tfl) / 0.04)
        j_model = (1 - g) * j_ohm + g * j_sclc
        loss = torch.mean((torch.log10(j_model) - torch.log10(jt)) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        mu = float(torch.exp(params[0]))
        n0 = float(torch.exp(params[1]))
        v_tfl = float(torch.exp(params[2]))
        m = 1 + 3 * float(torch.sigmoid(params[3]))
    return mu, n0, v_tfl, m


def fit_lsq_window(v, j, v_min=0.7):
    """传统做法A：人工选高压窗口对 V² 回归。返回 (mu 或 nan)。"""
    m = v >= v_min
    if m.sum() < 2:
        return np.nan
    slope = np.polyfit(v[m]**2, j[m], 1)[0]
    return 8 * slope * L**3 / (9 * EPS_R * EPS0)


def fit_lsq_all(v, j):
    """传统做法B：全点对 V² 回归（不选窗口，欧姆区点混入造成偏差）。"""
    slope = np.polyfit(v**2, j, 1)[0]
    return 8 * slope * L**3 / (9 * EPS_R * EPS0)


# %%
