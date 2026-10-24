# -*- coding: utf-8 -*-
"""
项目14：PINN 拟合 SCLC 迁移率 —— 第 1 部分：物理约束参数辨识
================================================================
核心思想（对应项目表"解决 SCLC 数据点少、小样本迁移率稳健估计"）：

  Mott-Gurney 律 J = (9/8)·εε₀·μ·V²/L³ 只在 SCLC 区成立，欧姆区 J ∝ V。
  传统做法要"人工选窗口"再对 V² 回归——点少时窗口里可能凑不出 2 个点；
  不选窗口则欧姆区点混入 → 系统性偏差；有陷阱时指数偏离 2，偏差更大。

  物理约束拟合的做法：把"欧姆区+平滑过渡+SCLC 区"整条物理曲线作为模型，
  一次性拟合全部数据点（不管落在哪个区），共享同一组物理参数
  (μ, n₀, V_TFL, m)。点少、有噪声时依然稳健，且不需要人工选窗口。

本脚本三件事：
  1. 用 Mott-Gurney + Lampert 理论生成合成 SCLC 曲线（真值 μ 已知）
  2. 蒙特卡洛对比：物理约束拟合 vs 两种传统 V² 拟合（各 300 次，
     干净场景 m=2 与陷阱场景 m=3 各做一遍）
  3. 套真实数据（B-3 批 4 号样品）看模型给出的物理诊断
"""

# %%
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)
np.random.seed(0)

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
# ------------------------- 1. 合成曲线总览图 -------------------------
v_dense = np.linspace(0.02, V_MAX, 200)

fig, ax = plt.subplots(figsize=(7, 5))
for m_true, style in ((2.0, "-"), (3.0, "--")):
    j_dense = sclc_curve(v_dense, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=m_true)
    ax.loglog(v_dense, j_dense, style, lw=2,
              label=f"真值曲线 m={m_true:.0f}" + ("（Mott-Gurney）" if m_true == 2 else "（指数陷阱）"))
ax.loglog(v_dense, Q * MU_TRUE * N0_TRUE * v_dense / L, ":", color="orange", label="欧姆延拓 (∝V)")
ax.loglog(v_dense, (9 / 8) * EPS_R * EPS0 * MU_TRUE * v_dense**2 / L**3, ":",
          color="green", label="Mott-Gurney (∝V²)")
ax.axvline(V_TFL_TRUE, ls="--", color="gray", label=f"V_TFL ≈ {V_TFL_TRUE:.2f} V")
ax.set_xlabel("Voltage (V)"); ax.set_ylabel("J (A/m²)")
ax.set_title(f"合成 SCLC 曲线（L=20nm, εr=3, μ*={MU_TRUE*1e4:.1e} cm²/V·s）")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig("synthetic_curve.png", dpi=150)
plt.close(fig)
print(f"V_TFL = {V_TFL_TRUE:.3f} V，合成曲线图已保存")


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
# ------------------------- 3. 蒙特卡洛：小样本稳健性 -------------------------
N_POINTS, N_TRIALS, NOISE = 8, 300, 0.05
rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, m_true in zip(axes, (2.0, 3.0)):
    j_dense = sclc_curve(v_dense, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=m_true)
    errs = {"物理约束拟合": [], "窗口V²(V≥0.7)": [], "全点V²": []}
    n_window_fail = 0
    for _ in range(N_TRIALS):
        idx = rng.choice(len(v_dense) - 5, N_POINTS, replace=False) + 5
        v_s = v_dense[idx]
        j_s = j_dense[idx] * (1 + rng.normal(0, NOISE, N_POINTS))
        mu_p = fit_physics(v_s, j_s)[0]
        mu_w = fit_lsq_window(v_s, j_s)
        if np.isnan(mu_w):
            n_window_fail += 1
        errs["物理约束拟合"].append(mu_p / MU_TRUE)
        errs["窗口V²(V≥0.7)"].append(mu_w / MU_TRUE)
        errs["全点V²"].append(fit_lsq_all(v_s, j_s) / MU_TRUE)

    # 每种方法各自剔除 NaN（窗口法凑不够点的轮次）再画箱线图
    data = [np.array(errs[k])[~np.isnan(errs[k])] * 100 for k in errs]
    bp = ax.boxplot(data, tick_labels=errs.keys(), showfliers=False,
                    patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#aed6f1")
    ax.axhline(100, color="red", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"真值 m={m_true:.0f}" +
                 (f"（窗口法 {n_window_fail}/{N_TRIALS} 轮凑不够点，未计入）"
                  if n_window_fail else ""))
    for k, e in zip(errs, data):
        e2 = e[~np.isnan(e)]
        print(f"m={m_true:.0f} {k:14s} 中位 {np.median(e2):6.1f}%  "
              f"IQR [{np.percentile(e2,25):5.1f}%, {np.percentile(e2,75):5.1f}%]")
    print()

axes[0].set_ylabel("μ 估计 / 真值 (%)")
fig.suptitle(f"小样本稳健性：每轮随机 {N_POINTS} 点 + {int(NOISE*100)}% 噪声，"
             f"{N_TRIALS} 轮（虚线=真值）")
fig.tight_layout()
fig.savefig("montecarlo_boxplot.png", dpi=150)
plt.close(fig)
print("蒙特卡洛对比图已保存")


# %%
# ------------------------- 4. 真实数据诊断：B-3 批 4 号 -------------------------
def load_instrument_csv(path):
    """解析源表导出文本 CSV（与项目10流水线同格式），返回 V(V), J(A/m²)。"""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    area = 0.1
    for ln in lines[:14]:
        if ln.startswith("Area (cm^2)"):
            area = float(ln.split(",")[1])
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("[Raw Data]"))
    v, j = [], []
    for ln in lines[idx + 1:]:
        p = ln.split(",")
        if len(p) < 3:
            break
        try:
            v.append(float(p[1])); j.append(float(p[2]) * 1e-3 / area * 1e4)
        except ValueError:
            break
    return np.array(v), np.array(j)


if not os.path.exists("data/b3_4.csv"):
    print("[提示] 未找到 data/b3_4.csv，跳过真实数据诊断——"
          "把你自己的 SCLC 导出文件放到 data/b3_4.csv 后重跑本脚本即可。")
    raise SystemExit(0)
v_real, j_real = load_instrument_csv("data/b3_4.csv")
pos = (v_real > 0.05) & (j_real > 0)
v_r, j_r = v_real[pos], j_real[pos]

# 小样本场景：只随机抽 10 个点（模拟"测得少"），加 3% 噪声
idx = np.linspace(0, len(v_r) - 1, 10).astype(int)
j_sub = j_r[idx] * (1 + np.random.default_rng(1).normal(0, 0.03, 10))
mu_r, n0_r, vt_r, m_r = fit_physics(v_r[idx], j_sub)

print("真实数据 B-3/4 wy：物理约束拟合诊断")
print(f"  指数 m = {m_r:.2f}   （m→1 欧姆主导；m=2 纯 Mott-Gurney；m>2 陷阱）")
if m_r < 1.6:
    print("  诊断：m 明显小于 2，曲线以欧姆区为主，未进入 SCLC 区——")
    print("        与项目10流水线判定一致，μ 不可辨识，不予报告。")
else:
    print(f"  μ = {mu_r*1e4:.2e} cm²/V·s（SnO₂ 典型 1e-3~1e-1 cm²/V·s，可对比）")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(v_r, j_r, "o", ms=4, label="实测 |J| 全部点")
ax.loglog(v_r[idx], j_sub, "s", ms=10, mfc="none", label="拟合用的 10 个抽样点")
v_line = np.linspace(0.05, V_MAX, 150)
ax.loglog(v_line, sclc_curve(v_line, mu_r, n0_r, vt_r, m=m_r), lw=2,
          label=f"物理约束拟合 (m={m_r:.2f})")
ax.set_xlabel("Voltage (V)"); ax.set_ylabel("J (A/m²)")
ax.set_title("真实 B-3/4 wy 曲线的物理约束拟合诊断")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig("real_b3_4_fit.png", dpi=150)
plt.close(fig)
print("真实数据诊断图已保存")
