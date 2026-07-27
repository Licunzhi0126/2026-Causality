# Native V7 × WYT 真实数据实验结果包

## 最终结论

旧 `wyt_cg_regsim_v7` 并不是完整 native V7：它用 H5AD Regulon/Module 活性 `R` 替代真实 GRN 状态 `G`，并直接池化 spot 特征。11.5→12.5 上，它自己的 ΔEI 为 `-1.917`；同一 assignment 用严格 native V7 重评只有 `+0.034`。

恢复真实 spot CCI+GRN 后，旧 pooled 宏观接口在 300 epoch 的严格 ΔEI 为：

- 11.5→12.5：`+0.649`
- 12.5→13.5：`+0.219`
- 13.5→14.5：`+0.110`

但训练代理和严格复算错位，1500 epoch 反而降为：`+0.583 / -0.054 / +0.066`。

新增 `macro_feature_mode=recompute_g` 后，每个 epoch 先池化真实表达，再用真实 GRN 重算 macro G。300 epoch 的严格 ΔEI 为：

- 11.5→12.5：`+1.675`
- 12.5→13.5：`+1.448`
- 13.5→14.5：`+2.786`

1500 epoch 提升到 `+2.197 / +1.756 / +2.971`，但 Keff 只有约 `8–9`，原型塌缩加重。

11.5→12.5 加强簇用量约束后，hardK 恢复到 `55/57`、Keff 为 `49.1/45.9`，严格 ΔEI 仍为 `+1.295`。因此完整适配后的高 ΔEI 并非完全依赖塌缩，但正式版本不应只追求无约束最高分。

## 重要方法学发现

- 自然 Seurat K40 三组 ΔEI 为 `0.918 / 0.666 / 0.817`；K150 为 `0.125 / -0.065 / 0.297`。
- 当前 V7 的 N-only ΔEI 约为 `-0.004`，G-only 与 N+G 几乎相同；最终 EI 数值主要由真实 GRN G 驱动。
- CCI 仍用于 N、图编码和宏观 CCI，但其最终 PIJ 校正尺度很弱。论文中若强调 CCI+GRN 联合驱动，需要继续校准 N/G 尺度。

## 文件说明

- `summaries/experiment_master_table.csv`：所有主要 WYT 运行的统一表。
- `summaries/recompute_g_300_summary.csv`：推荐接口的三时间对结果。
- `summaries/recompute_g_1500_summary.csv`：长训练与塌缩结果。
- `summaries/recompute_g_usage_constraint_comparison.csv`：11.5 抗塌陷对比。
- `summaries/feature_block_ablation.csv`：N/G 特征块消融。
- `selected_runs/`：关键原始 `all_results.csv` 与旧 RegSim 重评 JSON。
- `figures/`：报告中使用的图。

当前正式训练为单 seed=42；进入论文主表前仍需多 seed、强用量约束三时间对、K32/K40/K64 敏感性分析。
