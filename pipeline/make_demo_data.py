# -*- coding: utf-8 -*-
"""
生成 SCLC 流水线的演示数据（data/demo/）。
三个合成样品，对应三种典型情况：
  demo1_ohmic.csv      全程欧姆区（斜率≈1）→ 流水线判"无 SCLC 区"
  demo2_sclc.csv       欧姆区→SCLC 区完整转变 → 自动提取迁移率
  demo3_transition.csv 转变中（斜率介于 1 和 2 之间）→ 判"疑似"
运行：python make_demo_data.py
"""

# %%
import os
import numpy as np

Q = 1.602e-19
EPS0 = 8.854e-12
EPS_R = 3.0
L = 500e-9          # 演示器件厚度 500 nm
AREA = 0.1          # cm²


def curve(v, mu, n0, width=0.03):
    """欧姆区 + 平滑过渡 + Mott-Gurney SCLC（与 02-pinn-sclc 同一物理模型）。"""
    v_tfl = Q * n0 * L**2 / (2 * EPS_R * EPS0)
    j_ohm = Q * mu * n0 * v / L                      # A/m²
    j_sclc = (9 / 8) * EPS_R * EPS0 * mu * v**2 / L**3
    g = 1 / (1 + np.exp(-(v - v_tfl) / width))
    return (1 - g) * j_ohm + g * j_sclc              # A/m²


def write_instrument_csv(path, v, j_am2, rng):
    """按源表导出格式写文件（流水线 parse_instrument_csv 可直接解析）。"""
    i_ma = j_am2 * 1e-4 * AREA * 1e3                 # A/m² -> A -> mA
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("Name,TestIV 4.CH_REF(SCLC),,,,,,,,,,,,,\r\n")
        f.write("Note,demo data,,,,,,,,,,,,,,\r\n")
        f.write("Time Stamp,2026/1/1 12:00:00,,,,,,,,,,,,,\r\n")
        f.write("Duration (s),13.96,,,,,,,,,,,,,\r\n")
        f.write("[Parameter],,,,,,,,,,,,,,\r\n")
        f.write(f"Start (V),{v[0]:.6f},,,,,,,,,,,,,\r\n")
        f.write(f"End (V),{v[-1]:.6f},,,,,,,,,,,,,\r\n")
        f.write("Step (V),0.010000,,,,,,,,,,,,,\r\n")
        f.write(f"Area (cm^2),{AREA:.6f},,,,,,,,,,,,,\r\n")
        f.write("SUN,1.000000,,,,,,,,,,,,,\r\n")
        f.write("NPLC,1.000000,,,,,,,,,,,,,\r\n")
        f.write("Compliance (A),1.000000,,,,,,,,,,,,,\r\n")
        f.write("Sense mode,False,,,,,,,,,,,,,\r\n")
        f.write("degC,0,,,,,,,,,,,,,\r\n")
        f.write("[Result],Volt (V),I (mA),J (mA/cm^2),,,,,,,,,,,\r\n")
        f.write(f"[Raw Data],{v[0]:.15f},{i_ma[0]:.15f},{i_ma[0] / AREA:.15f},,,,,,,,,,,\r\n")
        for vi, ii in zip(v[1:], i_ma[1:]):
            f.write(f",{vi:.15f},{ii:.15f},{ii / AREA:.15f},,,,,,,,,,,\r\n")
    print("written", path)


os.makedirs("data/demo", exist_ok=True)
v = np.round(np.arange(-0.10, 1.301, 0.01), 2)
rng = np.random.default_rng(0)
# 真值：μ = 1e-3 cm²/V·s；n0 决定 V_TFL 位置（欧姆区/SCLC 区分界）
# V_TFL = q·n0·L²/2εε₀，L=500nm 时：n0=2e20 → V_TFL≈0.15V；6e20 → ≈0.45V；1e23 → ≈75V
samples = {
    "demo1_ohmic": dict(mu=1e-7, n0=1e23),        # V_TFL≈75V → 1.3V 内全程欧姆区
    "demo2_sclc": dict(mu=1e-7, n0=2.0e20),       # V_TFL≈0.15V → 高压段完整 SCLC 区
    "demo3_transition": dict(mu=1e-7, n0=1.2e21), # V_TFL≈0.9V → 扫描末端转变中
}
for name, kw in samples.items():
    j = curve(np.clip(v, 0.01, None), **kw)
    j = np.where(v > 0, j, abs(j) * 0.01 + 1e-6)   # 负压支给个小的对称电流
    j = j * (1 + rng.normal(0, 0.02, len(v)))      # 2% 噪声
    write_instrument_csv(os.path.join("data", "demo", name + ".csv"),
                         v, np.abs(j), rng)
print("\n完成。下一步：python sclc_pipeline.py（CONFIG 里 input_dirs 已指向 data/demo）")
