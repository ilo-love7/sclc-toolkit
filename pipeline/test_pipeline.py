# -*- coding: utf-8 -*-
"""SCLC 流水线单元测试：python -m pytest test_pipeline.py -v

覆盖四类关键行为：
  1. 解析器能读仪器导出格式（含真实验证过的 demo 文件）
  2. log-log 斜率在纯 V² 数据上恢复 2.0
  3. Mott-Gurney 拟合在无噪合成曲线上恢复真值迁移率
  4. 判定逻辑：欧姆区拒绝出数 / SCLC 区给出迁移率 / 转变中提示复测
运行前先生成演示数据：python make_demo_data.py
"""

# %%
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sclc_pipeline import (analyze_sample, loglog_slope,
                           maybe_mobility, mott_gurney_mobility,
                           parse_instrument_csv)

HERE = os.path.dirname(os.path.abspath(__file__))
Q, EPS0, EPS_R, L, AREA = 1.602e-19, 8.854e-12, 3.0, 500e-9, 0.1
MU_TRUE = 1e-7          # m²/V·s（make_demo_data 的真值）


@pytest.fixture(scope="session", autouse=True)
def demo_data():
    """确保演示数据存在（没有就用脚本生成一次）。"""
    if not os.path.exists(os.path.join(HERE, "data", "demo", "demo2_sclc.csv")):
        subprocess.run([sys.executable, "make_demo_data.py"], check=True,
                       cwd=HERE, capture_output=True)


# %%
def test_parse_instrument_csv():
    meta, v, i = parse_instrument_csv(
        os.path.join(HERE, "data", "demo", "demo2_sclc.csv"))
    assert len(v) == 141                      # -0.1..1.3 步长 0.01
    assert meta["Area (cm^2)"] == pytest.approx(0.1)
    assert v[0] == pytest.approx(-0.1) and v[-1] == pytest.approx(1.3)
    assert np.all(np.isfinite(i))


def test_loglog_slope_recovers_v2():
    v = np.linspace(0.1, 1.3, 50)
    j = 3.0 * v**2
    assert loglog_slope(v, j, 0.2, 1.2) == pytest.approx(2.0, abs=0.01)


def test_loglog_slope_recovers_ohmic():
    v = np.linspace(0.1, 1.3, 50)
    j = 5.0 * v
    assert loglog_slope(v, j, 0.2, 1.2) == pytest.approx(1.0, abs=0.01)


def test_mott_gurney_recovers_mobility():
    """无噪 SCLC 曲线上迁移率反演误差应 <1%。"""
    v = np.linspace(0.2, 1.3, 100)
    j = (9 / 8) * EPS_R * EPS0 * MU_TRUE * v**2 / L**3 / 1e4   # A/cm²（流水线约定）
    mu, r = mott_gurney_mobility(v, j, L, EPS_R)
    assert r > 0.999
    assert mu == pytest.approx(MU_TRUE * 1e4, rel=0.01)   # cm²/V·s（函数已换算）


# %%
def test_verdict_ohmic_refuses_mobility():
    meta, v, i = parse_instrument_csv(
        os.path.join(HERE, "data", "demo", "demo1_ohmic.csv"))
    res = analyze_sample(meta, v, i)
    assert "欧姆区主导" in res["verdict"]
    assert np.isnan(res["mobility_cm2Vs"])    # 必须拒绝出数


def test_verdict_sclc_extracts_mobility():
    meta, v, i = parse_instrument_csv(
        os.path.join(HERE, "data", "demo", "demo2_sclc.csv"))
    res = analyze_sample(meta, v, i)
    maybe_mobility(res, 0.5, 3.0)   # 演示膜厚 500 nm，εr=3
    assert "SCLC 区已出现" in res["verdict"]
    assert res["k2"] == pytest.approx(2.0, abs=0.1)
    # 反演迁移率应接近真值 1e-3 cm²/V·s（容差 2 倍以内）
    assert res["mobility_cm2Vs"] == pytest.approx(MU_TRUE * 1e4, rel=1.0)


def test_verdict_transition_suggests_rescan():
    meta, v, i = parse_instrument_csv(
        os.path.join(HERE, "data", "demo", "demo3_transition.csv"))
    res = analyze_sample(meta, v, i)
    assert "疑似" in res["verdict"]
    assert np.isnan(res["mobility_cm2Vs"])
