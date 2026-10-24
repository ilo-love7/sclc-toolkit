# -*- coding: utf-8 -*-
"""
项目14 第 2 部分：DeepXDE 逆问题——直接从物理方程辨识迁移率
================================================================
物理模型（电子-only 器件，SCLC 区）：
    泊松方程   d²φ/dx² = q·n/εε₀
    漂移电流   J = q·μ·n·(dφ/dx) = μ·εε₀·φ''·φ'   （J 沿 x 为常数）
  消去 n，J 为常数的条件：d/dx(φ''·φ') = 0

归一化 ξ=x/L, u=φ/V：残差 R = u'''·u' + (u'')² = 0，
边界 u(0)=0, u(1)=1, u'(0)=0（第三个条件必不可少，否则 u''u'=c 中
c 自由，μ 与形状互相补偿，逆问题欠定）。

【解析基参数化】直接用普通网络拟合 u=ξ^1.5 时，tanh 网络在 ξ→0 的
形状误差会全部被 μ 吸收（实测偏差 10 倍以上）——这是 PINN 的经典
刚度问题。标准解法是把已知解析结构写进参数化：
    u(ξ) = ξ^(3/2) · g(ξ)，网络只学修正因子 g（真值 g≡1）
u(0)=0 与 u'(0)=0 被结构自动满足，边界条件只剩 g(1)=1，
μ 只吸收 g 的高阶偏差，辨识精度大幅提高。

数据进入方式：J(Vᵢ) = εε₀·μ·Vᵢ²/L³ · u'(1)·u''(1)（ξ=1 边界）。
"""

# %%
import os
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import deepxde as dde

dde.config.set_default_float("float64")

# ------------------------- 物理常数与合成"测量"数据 -------------------------
EPS0 = 8.854e-12
L = 20e-9
EPS_R = 3.0

MU_TRUE = 2e-7      # 真值迁移率 m²/V·s (=2e-3 cm²/V·s)
V_MAX = 1.3

rng = np.random.default_rng(7)
V_MEAS = np.sort(rng.uniform(0.4, V_MAX, 8))
J_MEAS = (9 / 8) * EPS_R * EPS0 * MU_TRUE * V_MEAS**2 / L**3 \
    * (1 + rng.normal(0, 0.05, 8))
print("合成测量点 (V, J):")
for v_, j_ in zip(V_MEAS, J_MEAS):
    print(f"  {v_:.2f} V  {j_:.3e} A/m²")


# %%
# ------------------------- 网络、残差、边界条件 -------------------------
net = dde.maps.FNN([2] + [48] * 3 + [1], "tanh", "Glorot uniform")

log_mu = dde.Variable(np.log(2e-8))   # 初值比真值低一个量级，考验收敛


def make_u(x, out):
    """解析基合成：u(ξ) = ξ^(3/2)·g(ξ)。返回合成后的 u 张量。"""
    xi = x[:, 0:1]
    return xi ** 1.5 * out[:, 0:1]


def pde_residual(x, out):
    u = make_u(x, out)
    u_prime = dde.grad.jacobian(u, x, j=0)
    u_2 = dde.grad.hessian(u, x, i=0, j=0)
    u_3 = dde.grad.jacobian(u_2, x, j=0)
    r = u_3 * u_prime + u_2**2
    # ξ→0 处残差按 1/ξ 发散（可积），乘 ξ² 压制奇异区权重
    return [(x[:, 0:1] ** 2) * r]


# 解析基在 ξ=0 处 u''∝ξ^(-1/2) 发散（0×inf=NaN），配点域下界取 0.01 避开；
# u(0)=0、u'(0)=0 由解析基结构自动满足，域内无需 ξ=0 的任何约束点
geom = dde.geometry.Rectangle([0.01, 0.0], [1.0, 1.0])


def boundary_xi1(x, on):
    return on and np.isclose(x[0], 1.0)


# u(0)=0 与 u'(0)=0 由解析基结构自动满足，只剩 g(1)=1（即 u(1)=1）
bc_g1 = dde.icbc.DirichletBC(geom, lambda x: np.ones((len(x), 1)),
                             boundary_xi1, component=0)


def data_residual(inputs, out, X):
    """实测 J-V 从 ξ=1 边界进入。求导必须用张量 inputs。"""
    import torch
    u = make_u(inputs, out)
    up = dde.grad.jacobian(u, inputs, j=0)
    u2 = dde.grad.hessian(u, inputs, i=0, j=0)
    mu = torch.exp(log_mu)
    v = inputs[:, 1:2] * V_MAX
    j_pred = EPS_R * EPS0 * mu * v**2 / L**3 * up * u2
    v_np = v.detach().cpu().numpy().ravel()
    j_meas = np.interp(v_np, V_MEAS, J_MEAS).reshape(-1, 1)
    j_meas = torch.as_tensor(j_meas, dtype=torch.float64)
    return (j_pred - j_meas) / j_meas


data_bc = dde.icbc.OperatorBC(geom, data_residual, boundary_xi1)

data = dde.data.PDE(geom, pde_residual, [bc_g1, data_bc],
                    num_domain=1200, num_boundary=400)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3,
              loss_weights=[5, 10, 10],
              external_trainable_variables=[log_mu])

PERIOD = 1000
variable = dde.callbacks.VariableValue(log_mu, period=PERIOD)
losshistory, train_state = model.train(iterations=8000, callbacks=[variable],
                                       display_every=2000)

# %%
# ------------------------- 结果检查 -------------------------
mu_hist = np.exp(np.array(variable.value)).ravel()
mu_fit = mu_hist[-1]
print("\n================ 逆问题结果 ================")
print(f"真值 μ    = {MU_TRUE*1e4:.2e} cm²/V·s")
print(f"PINN 辨识 = {mu_fit*1e4:.2e} cm²/V·s  "
      f"(相对误差 {abs(mu_fit/MU_TRUE-1)*100:.1f}%)")

xi = np.linspace(0.05, 1.0, 50).reshape(-1, 1)
u_pred = model.predict(np.hstack([xi, np.full_like(xi, 0.8)]))[:, 0]
shape_err = np.max(np.abs(u_pred - xi[:, 0] ** 1.5))
print(f"解形状 vs 解析解 ξ^1.5 最大偏差: {shape_err:.4f}")

print("""
---------------- 结论与局限 ----------------
1. 框架本身成立：方程已作为 PDE 残差嵌入，μ 作为可训练变量随数据收敛，
   全流程（前向解 BVP + 逆问题辨识）可运行、可复现。
2. 但无陷阱 Mott-Gurney 问题的 PDE 逆问题是【退化】的：
   - 该边值问题存在唯一解析解 u=ξ^1.5（即 c=u''u'≡9/8 由边界条件唯一
     决定），μ 辨识在数学上等价于代数运算 → 第 1 部分（pinn_sclc.py）
     的物理约束参数辨识就是这个代数问题的稳健解法；
   - 允许网络自由拟合形状时，形状误差与 μ 互相补偿（欠定），
     实测偏差达数倍～数十倍，且加训练不能根除。
3. PDE-PINN 逆问题的真正用武之地：在方程中加入代数律没有的物理
   （陷阱动力学、扩散项、场致迁移率），使形状不再有解析解——
   那时形状由方程唯一决定，μ 才能被独立辨识。这需要复测的
   高电压 SCLC 数据（End V 提到 5–8 V）支撑，是项目14的后续工作。
""")

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

steps = np.arange(len(mu_hist)) * PERIOD
np.savetxt("mu_history.csv",
           np.column_stack([steps, mu_hist]), delimiter=",",
           header="step,mu_m2Vs", comments="")
# μ 历史多于 2 点画收敛曲线；否则画"真值 vs 辨识值"对比条形图
if len(mu_hist) >= 3:
    axes[0].plot(steps, mu_hist * 1e4, lw=2)
    axes[0].set_xlabel("训练步数")
    axes[0].set_title("PINN 逆问题：迁移率收敛过程")
else:
    names = ["真值", "PINN 辨识"]
    vals = [MU_TRUE * 1e4, mu_fit * 1e4]
    bars = axes[0].bar(names, vals, color=["#d62728", "#1f77b4"], width=0.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("μ (cm²/V·s)")
    axes[0].set_title("PINN 逆问题：辨识值 vs 真值")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1e}",
                     ha="center", fontsize=10)
    axes[0].legend()
axes[0].set_ylabel("μ (cm²/V·s)")

v_line = np.linspace(0.4, V_MAX, 100)
axes[1].loglog(V_MEAS, J_MEAS, "o", ms=8, label="合成测量点 (8点+5%噪声)")
axes[1].loglog(v_line, (9 / 8) * EPS_R * EPS0 * mu_fit * v_line**2 / L**3,
               "-", lw=2, label="PINN 辨识 μ 的 Mott-Gurney 曲线")
axes[1].loglog(v_line, (9 / 8) * EPS_R * EPS0 * MU_TRUE * v_line**2 / L**3,
               "--", lw=1.5, label="真值曲线")
axes[1].set_xlabel("Voltage (V)"); axes[1].set_ylabel("J (A/m²)")
axes[1].set_title("稀疏数据 + PINN 逆问题 → J-V 重构")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig("deepxde_inverse.png", dpi=150)
print("结果图已保存 deepxde_inverse.png")
