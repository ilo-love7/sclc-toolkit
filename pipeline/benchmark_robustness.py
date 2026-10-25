# -*- coding: utf-8 -*-
"""
SCLC 判定逻辑噪声鲁棒性基准
============================
问题：项目 README 声称"数据不达标就拒绝出数"是可靠的设计——但判定逻辑
本身能容忍多大的测量噪声？

方法：用同一物理模型生成"真值已知"的欧姆型与 SCLC 型曲线，扫描噪声水平
（2%~30%），每个水平重复 200 次，统计三种判定（欧姆拒绝 / SCLC 提取 /
转变中提示）的准确率。

产出：figs/fig_noise_robustness.png + 控制台准确率表。
"""

# %%
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sclc_pipeline import analyze_sample, maybe_mobility

Q = 1.602e-19
EPS0 = 8.854e-12
EPS_R = 3.0
L = 500e-9
AREA = 0.1


def curve(v, mu, n0, width=0.03):
    """欧姆区 + 平滑过渡 + SCLC（与 make_demo_data 同一模型）。"""
    v_tfl = Q * n0 * L**2 / (2 * EPS_R * EPS0)
    j_ohm = Q * mu * n0 * v / L
    j_sclc = (9 / 8) * EPS_R * EPS0 * mu * v**2 / L**3
    g = 1 / (1 + np.exp(-(v - v_tfl) / width))
    return (1 - g) * j_ohm + g * j_sclc


def make_sample(mu, n0, noise, rng):
    """生成一条带噪暗态 J-V，并包装成 analyze_sample 需要的输入。"""
    v = np.round(np.arange(-0.10, 1.301, 0.01), 2)
    j = curve(np.clip(v, 0.01, None), mu, n0)
    j = np.where(v > 0, j, np.abs(j) * 0.01 + 1e-6)
    j = j * (1 + rng.normal(0, noise, len(v)))
    i = np.abs(j) * 1e-4 * AREA * 1e3          # A/m² -> mA
    meta = {"Area (cm^2)": AREA}
    return meta, v, i


# 判定是否正确：欧姆/转变中 → 应拒绝出数；SCLC → 应给出迁移率
def verdict_correct(res, expect_sclc):
    """SCLC 样品：应给出迁移率；欧姆样品：应拒绝（nan）。"""
    gave_mu = not np.isnan(res["mobility_cm2Vs"])
    return gave_mu == expect_sclc


# %%
NOISE_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_TRIALS = 200
rng = np.random.default_rng(0)

acc = {"SCLC 型样品（应提取）": [], "欧姆型样品（应拒绝）": []}
for noise in NOISE_LEVELS:
    correct = {"SCLC 型样品（应提取）": 0, "欧姆型样品（应拒绝）": 0}
    for kind, (mu, n0) in (("SCLC 型样品（应提取）", (1e-7, 2.0e20)),
                           ("欧姆型样品（应拒绝）", (1e-7, 1e23))):
        for _ in range(N_TRIALS):
            meta, v, i = make_sample(mu, n0, noise, rng)
            res = analyze_sample(meta, v, i)
            maybe_mobility(res, L * 1e6, EPS_R)   # 演示膜厚 0.5 μm
            correct[kind] += verdict_correct(res, expect_sclc=(kind.startswith("SCLC")))
    for k in acc:
        acc[k].append(correct[k] / N_TRIALS)

print(f"{'噪声水平':>8s} {'SCLC判定正确率':>14s} {'欧姆拒绝正确率':>14s}")
for noise, a1, a2 in zip(NOISE_LEVELS, acc["SCLC 型样品（应提取）"],
                         acc["欧姆型样品（应拒绝）"]):
    print(f"{noise*100:7.0f}% {a1*100:13.1f}% {a2*100:13.1f}%")

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot([n * 100 for n in NOISE_LEVELS],
        [a * 100 for a in acc["SCLC 型样品（应提取）"]], "o-", lw=2,
        label="SCLC 型：正确提取迁移率")
ax.plot([n * 100 for n in NOISE_LEVELS],
        [a * 100 for a in acc["欧姆型样品（应拒绝）"]], "s--", lw=2,
        label="欧姆型：正确拒绝出数")
ax.set_xlabel("测量噪声水平 (%)")
ax.set_ylabel("判定正确率 (%)")
ax.set_title(f"判定逻辑噪声鲁棒性（每点 {N_TRIALS} 次合成试验）")
ax.legend()
fig.tight_layout()
out = os.path.join("figs", "fig_noise_robustness.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("图已保存", out)
