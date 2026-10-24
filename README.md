# SCLC Toolkit · SCLC 载流子迁移率测量与分析工具箱

围绕 **空间电荷限制电流（SCLC）迁移率测量** 的两个可复现模块：
从仪器原始数据到迁移率数字的完整链路——先判定数据配不配，再谈提取。

作者：ilo · 数据与隐私：实验室原始数据不随仓库分发，仓库自带合成演示数据

| 模块 | 内容 | 核心结论 |
|---|---|---|
| [pipeline](pipeline/) | 仪器数据解析 → log-log 分区判定 → V_TFL 检测 → Mott-Gurney 迁移率 | 数据不达 SCLC 区时**拒绝**输出迁移率（8 个真实样品复核全部正确拒绝） |
| [pinn](pinn/) | 物理约束参数辨识 + DeepXDE 逆问题及病态性分析 | 陷阱场景下传统窗口 V² 法高估 52%，物理约束拟合仅偏 2%；无陷阱 Mott-Gurney 的 PDE 逆问题退化 |

## 为什么"拒绝出数"是特性不是缺陷

SCLC 迁移率只在 J-V 曲线进入 V² 区后才有物理意义。流水线用 log-log 斜率
做分区判定：欧姆区主导时明确报告"无 SCLC 区"并给出复测建议（提高扫描
电压），而不是硬套公式输出一个无意义的数字——错误数据链上的每个数字
都会向下游放大。

## 快速开始

```bash
# pipeline：合成演示数据三种判定，一键复现
cd pipeline && python make_demo_data.py && python sclc_pipeline.py

# pinn：合成数据蒙特卡洛 + 真实数据诊断接口
cd pinn && python pinn_sclc.py
```

依赖见根目录 `requirements.txt`（numpy / pandas / matplotlib / torch / deepxde）。

## License

MIT
