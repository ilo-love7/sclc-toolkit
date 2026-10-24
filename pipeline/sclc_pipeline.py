# -*- coding: utf-8 -*-
"""
SCLC 测量流水线（项目10：SCLC 测量流水线标准化）
================================================
输入：源表导出的暗态 J-V 数据文件夹（如 B-3、B-3(1)）
输出：
  results/summary.csv        —— 每个样品一行的分区判定 + 参数汇总表
  results/<样品名>_loglog.png —— log-log 判定图（欧姆区/SCLC 区/拟合线标出）
  控制台报告                  —— 中文逐样品结论

判定逻辑（对应 Mott-Gurney 框架）：
  在 log(J)-log(V) 上，
    欧姆区           斜率 ≈ 1（热生载流子导电）
    空间电荷限制区   斜率 ≈ 2（SCLC，J = 9εε₀μV²/8L³）
    陷阱填充极限电压  V_TFL：两段斜率之间的拐点
  若全程斜率 ≈ 1 且未出现拐点 → 判"无 SCLC 区"，拒绝硬算迁移率。
"""

# %%
import glob
import hashlib
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 自带微软雅黑，解决图中中文显示成方框的问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# %%
# ------------------------- 配置区：换数据只改这里 -------------------------
CONFIG = {
    # 要处理的文件夹（可加多个）；仓库自带 data/demo 演示数据（make_demo_data.py 生成）
    "input_dirs": ["data/demo"],
    # 结果输出文件夹
    "output_dir": "results",
    # 钙钛矿相对介电常数 εr
    # 实验方案给定：聚合物(PCBM) εr = 3（2026-10 实验方案 PPT）
    "epsilon_r": 3.0,
    # 膜厚/μm：未在 thickness_um 表中列出的样品用默认值，拿到实测值后按样品名覆盖。
    # 演示数据按 500 nm 生成；自己的器件请改成真实膜厚（μ ∝ 1/L³，必须准）。
    "thickness_um_default": 0.5,
    # 按样品名(小写)覆盖默认膜厚，例：{"b-3 / 1 wy": 0.022}
    "thickness_um": {},
}

# 判定阈值（一般不用改）
SLOPE_SCLC = 1.6    # 高压段斜率 ≥ 1.6 认为进入 SCLC 区
SLOPE_OHMIC_HI = 1.3  # 高压段斜率 ≤ 1.3 认为仍是欧姆区；介于两者之间 = 疑似转变


# %%
# ------------------------- 解析：仪器导出文件 → 数据 -------------------------
def parse_instrument_csv(path):
    """解析源表导出的文本 CSV（后缀可能是 xlsx，实为 ASCII 文本）。

    返回 (meta, V, I)。meta 是参数字典；V 单位 V，I 单位 A。
    文件结构：
      [Parameter] 区：Start (V), End (V), Step (V), Area (cm^2) ...
      [Raw Data] 后每行: ,电压, 电流(mA), 电流密度(mA/cm^2), ...
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    meta = {}
    for ln in lines[:14]:
        parts = ln.split(",")
        key = parts[0].strip()
        if key in ("Start (V)", "End (V)", "Step (V)", "Area (cm^2)",
                   "Duration (s)", "SUN", "Compliance (A)"):
            try:
                meta[key] = float(parts[1])
            except (ValueError, IndexError):
                pass

    raw_idx = next((i for i, ln in enumerate(lines)
                    if ln.startswith("[Raw Data]")), None)
    if raw_idx is None:
        raise ValueError("找不到 [Raw Data] 区，不是源表导出文件")

    volts, amps = [], []
    for ln in lines[raw_idx + 1:]:
        parts = ln.split(",")
        if len(parts) < 3:
            break
        try:
            volts.append(float(parts[1]))          # Volt (V)
            amps.append(float(parts[2]) * 1e-3)    # I (mA) -> A
        except ValueError:
            break
    return meta, np.array(volts), np.array(amps)


def collect_files(input_dirs):
    """收集各文件夹里的数据文件，按内容去重（导出常带同名副本）。"""
    seen, files = set(), []
    for d in input_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isfile(path):
                continue
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".txt", ".mia")):
                continue
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            files.append(path)
    return files


# %%
# ------------------------- 分析：分区判定 + 迁移率 -------------------------
def loglog_slope(v, j, lo, hi):
    """在 [lo, hi] V 窗口内拟合 log(J)-log(V) 斜率。"""
    m = (v >= lo) & (v <= hi) & (np.abs(j) > 0)
    if m.sum() < 5:
        return None
    k, _ = np.polyfit(np.log10(v[m]), np.log10(np.abs(j[m])), 1)
    return float(k)


def find_transition(v, j):
    """两段折线拟合找拐点（V_TFL 候选）与前后斜率。"""
    m = (v > 0.1) & (np.abs(j) > 0)
    lv = np.log10(v[m])
    lj = np.log10(np.abs(j[m]))
    best = None
    for i in range(5, len(lv) - 5):
        k1 = np.polyfit(lv[:i], lj[:i], 1)
        k2 = np.polyfit(lv[i:], lj[i:], 1)
        rss = (np.sum((lj[:i] - np.polyval(k1, lv[:i])) ** 2)
               + np.sum((lj[i:] - np.polyval(k2, lv[i:])) ** 2))
        if best is None or rss < best["rss"]:
            best = {"rss": rss, "v_kink": 10 ** lv[i],
                    "k1": float(k1[0]), "k2": float(k2[0])}
    return best


def mott_gurney_mobility(v, j, thickness_m, eps_r):
    """在 SCLC 窗口内拟合 J = (9/8)·εε₀·μ·V²/L³，返回 μ (cm²/V·s) 与拟合区间。

    做法：J 对 V² 线性回归，斜率 s = (9/8)εε₀μ/L³ → μ = 8·s·L³ / (9εε₀)。
    """
    eps0 = 8.854e-12  # F/m
    v2 = v ** 2
    s, b = np.polyfit(v2, np.abs(j), 1)
    r = np.corrcoef(v2, np.abs(j))[0, 1]
    mu = 8.0 * s * thickness_m ** 3 / (9.0 * eps_r * eps0)
    return mu * 1e4, r  # m²/V·s -> cm²/V·s


def analyze_sample(meta, v, i):
    """单个样品的完整分析，返回结果字典。"""
    area = meta.get("Area (cm^2)", np.nan)
    j = i / area
    pos = v > 0
    vpos, jpos = v[pos], j[pos]

    vmax = v.max()
    windows = [(0.05, 0.3), (0.3, 0.7), (0.7, min(1.3, vmax))]
    slopes = {f"{lo:.2f}-{hi:.2f} V": loglog_slope(vpos, jpos, lo, hi)
              for lo, hi in windows}

    trans = find_transition(vpos, jpos)
    k2 = trans["k2"]

    if k2 >= SLOPE_SCLC:
        verdict = "SCLC 区已出现，可提取迁移率（需膜厚）"
    elif k2 > SLOPE_OHMIC_HI:
        verdict = "疑似转变中：高压段斜率介于欧姆与SCLC之间，建议提高扫描电压复测"
    else:
        verdict = "欧姆区主导，无 SCLC 区，不能提取迁移率"

    result = {
        "scan_V": f"{meta.get('Start (V)', np.nan):.2f}→{meta.get('End (V)', np.nan):.2f}",
        "area_cm2": area,
        "n_points": len(v),
        "Imax_A": float(np.abs(i).max()),
        "slope_low": slopes["0.05-0.30 V"],
        "slope_mid": slopes["0.30-0.70 V"],
        "slope_high": slopes[f"0.70-{min(1.3, vmax):.2f} V"],
        "V_kink_V": trans["v_kink"],
        "k1": trans["k1"],
        "k2": trans["k2"],
        "verdict": verdict,
        "mobility_cm2Vs": np.nan,
        "V2_fit_R": np.nan,
        "mobility_ref_cm2Vs": np.nan,
        "V2_fit_R_ref": np.nan,
        "_vpos": vpos,
        "_jpos": jpos,
    }
    return result


def maybe_mobility(result, thickness_um, eps_r):
    """迁移率计算：SCLC 区出现时算正式值；否则也算一列"参考拟合"。

    参考拟合按实验方案的做法（J 对 V² 回归）在 0.7 V 以上窗口强行拟合，
    但判定为非 SCLC 区时数值不可信，只用来和正常迁移率量级对比。
    """
    if thickness_um is None:
        return
    vpos, jpos = result["_vpos"], result["_jpos"]

    # 正式值：只用拐点之后的 SCLC 窗口
    if result["k2"] >= SLOPE_SCLC:
        m = vpos >= result["V_kink_V"]
        mu, r = mott_gurney_mobility(vpos[m], jpos[m],
                                     thickness_um * 1e-6, eps_r)
        result["mobility_cm2Vs"] = mu
        result["V2_fit_R"] = r

    # 参考值：固定取 0.7 V 以上窗口，无论判定结果如何都算
    m_ref = vpos >= 0.7
    if m_ref.sum() > 5:
        mu_ref, r_ref = mott_gurney_mobility(vpos[m_ref], jpos[m_ref],
                                             thickness_um * 1e-6, eps_r)
        result["mobility_ref_cm2Vs"] = mu_ref
        result["V2_fit_R_ref"] = r_ref


# %%
# ------------------------- 绘图：log-log 判定图 -------------------------
def plot_sample(result, sample_name, out_png):
    v, j = result["_vpos"], result["_jpos"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(v, np.abs(j), "o", ms=4, color="#c0392b", label="|J| 实测")
    ax.axvline(result["V_kink_V"], color="#2c3e50", ls="--", lw=1,
               label=f"拐点 V≈{result['V_kink_V']:.2f} V")

    # 标出两段拟合斜率
    m1 = v <= result["V_kink_V"]
    for seg, k in ((m1, result["k1"]), (~m1, result["k2"])):
        if seg.sum() > 3:
            lv = np.log10(v[seg])
            lj = np.polyval(np.polyfit(lv, np.log10(np.abs(j[seg])), 1), lv)
            ax.loglog(v[seg], 10 ** lj, "-", lw=2, color="#2980b9",
                      label=f"拟合段斜率 {k:.2f}")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("|J| (A/cm²)")
    ax.set_title(f"{sample_name}\n{result['verdict']}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# %%
# ------------------------- 批处理主流程 -------------------------
def run(config=CONFIG):
    os.makedirs(config["output_dir"], exist_ok=True)
    files = collect_files(config["input_dirs"])
    print(f"共找到 {len(files)} 个数据文件（已按内容去重）\n")

    rows, details = [], []
    for path in files:
        batch = os.path.basename(os.path.dirname(path))
        sample = f"{batch} / {os.path.splitext(os.path.basename(path))[0]}"
        try:
            meta, v, i = parse_instrument_csv(path)
            result = analyze_sample(meta, v, i)
        except Exception as e:
            print(f"[跳过] {sample}: {e}")
            continue
        result["sample"] = sample

        # 膜厚：先查按样品名的覆盖表，查不到用实验方案默认值(SnO₂≈20nm)
        thick = config["thickness_um"].get(sample.lower(),
                                           config.get("thickness_um_default"))
        maybe_mobility(result, thick, config["epsilon_r"])

        safe_name = re.sub(r'[\\/:*?"<>| ]', "_", sample)
        png = os.path.join(config["output_dir"], f"{safe_name}_loglog.png")
        plot_sample(result, sample, png)

        def fmt(mu, r):
            return f"{mu:.2e} (R={r:.3f})" if not np.isnan(mu) else "—"

        print(f"◆ {sample}")
        print(f"  扫描 {result['scan_V']} V，{result['n_points']} 点，"
              f"面积 {result['area_cm2']:.2f} cm²，Imax {result['Imax_A']:.2e} A")
        print(f"  log-log 斜率：低压 {result['slope_low']:.2f} / "
              f"中压 {result['slope_mid']:.2f} / 高压 {result['slope_high']:.2f}")
        print(f"  拐点 ≈ {result['V_kink_V']:.2f} V"
              f"（前段斜率 {result['k1']:.2f} → 后段 {result['k2']:.2f}）")
        print(f"  结论：{result['verdict']}")
        print(f"  迁移率(正式, 仅SCLC区)：{fmt(result['mobility_cm2Vs'], result['V2_fit_R'])}")
        print(f"  参考拟合(0.7V以上, 非SCLC区时不可信)："
              f"{fmt(result['mobility_ref_cm2Vs'], result['V2_fit_R_ref'])}")
        print(f"  判定图：{os.path.basename(png)}\n")

        rows.append({k: result[k] for k in
                     ("sample", "scan_V", "area_cm2", "n_points",
                      "slope_low", "slope_mid", "slope_high",
                      "V_kink_V", "k1", "k2", "verdict",
                      "mobility_cm2Vs", "V2_fit_R",
                      "mobility_ref_cm2Vs", "V2_fit_R_ref")})
        details.append((sample, result))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config["output_dir"], "summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"汇总表已保存：{csv_path}")
    return df, details


# %%
if __name__ == "__main__":
    run()
