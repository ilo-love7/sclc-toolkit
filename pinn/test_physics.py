# -*- coding: utf-8 -*-
"""pinn 模块单元测试：python -m pytest test_physics.py -v

覆盖：
  1. sclc_curve 的分区物理行为（欧姆区斜率≈1、SCLC 区斜率≈2、陷阱抬高指数）
  2. fit_physics 在小样本含噪数据上恢复真值迁移率
  3. 传统窗口 V² 拟合在陷阱场景下的系统性高估（README 结论的可测版本）
"""

# %%
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics import EPS0, EPS_R, L, MU_TRUE, N0_TRUE, V_TFL_TRUE, \
    fit_lsq_window, fit_physics, sclc_curve


# %%
def test_sclc_curve_ohmic_regime():
    """V << V_TFL：J ∝ V（log-log 斜率≈1）。"""
    v = np.linspace(0.01, 0.03, 20)          # V_TFL=0.3V，此段为欧姆区
    j = sclc_curve(v, MU_TRUE, N0_TRUE, V_TFL_TRUE)
    k = np.polyfit(np.log10(v), np.log10(j), 1)[0]
    assert k == pytest.approx(1.0, abs=0.1)


def test_sclc_curve_sclc_regime():
    """V >> V_TFL、m=2：J ∝ V²（Mott-Gurney）。"""
    v = np.linspace(0.7, 1.3, 20)
    j = sclc_curve(v, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=2.0)
    k = np.polyfit(np.log10(v), np.log10(j), 1)[0]
    assert k == pytest.approx(2.0, abs=0.1)


def test_sclc_curve_trap_exponent():
    """陷阱场景 m=3：高压段斜率应为 3。"""
    v = np.linspace(0.7, 1.3, 20)
    j = sclc_curve(v, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=3.0)
    k = np.polyfit(np.log10(v), np.log10(j), 1)[0]
    assert k == pytest.approx(3.0, abs=0.1)


def test_v_tfl_formula():
    """V_TFL 与 Lampert 公式一致：q·n₀·L²/2εε₀。"""
    q = 1.602e-19
    expected = q * N0_TRUE * L**2 / (2 * EPS_R * EPS0)
    assert V_TFL_TRUE == pytest.approx(expected, rel=1e-9)


# %%
def test_fit_physics_recovers_mu_small_sample():
    """10 个点 + 3% 噪声：物理约束拟合恢复 μ（容差 25%）。"""
    rng = np.random.default_rng(7)
    v = np.sort(rng.uniform(0.05, 1.3, 10))
    j = sclc_curve(v, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=2.0) \
        * (1 + rng.normal(0, 0.03, 10))
    mu, n0, v_tfl, m = fit_physics(v, j)
    assert mu == pytest.approx(MU_TRUE, rel=0.25)
    assert 1.7 < m < 2.4


def test_lsq_window_biased_under_traps():
    """陷阱场景（m=3）：窗口 V² 拟合系统性高估 μ（README 52% 结论的方向）。"""
    rng = np.random.default_rng(3)
    v = np.sort(rng.uniform(0.4, 1.3, 8))
    j = sclc_curve(v, MU_TRUE, N0_TRUE, V_TFL_TRUE, m=3.0) \
        * (1 + rng.normal(0, 0.05, 8))
    mu_naive = fit_lsq_window(v, j, v_min=0.7)
    mu_phys = fit_physics(v, j)[0]
    assert mu_naive > MU_TRUE * 1.2          # 传统法显著高估
    assert mu_phys < mu_naive                # 物理约束拟合更接近真值
    assert mu_phys == pytest.approx(MU_TRUE, rel=0.35)
