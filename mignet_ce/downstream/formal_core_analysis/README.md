# Formal core analysis — 独立的正式论文结果整理/绘图流水线

**一次命令输出核心图表**：`python -m mignet_ce.downstream.formal_core_analysis ...`。
不 import 旧 `downstream/paper_assets`。沿用原 paper_assets Figure 1/2 的视觉布局与配色；Figure 2 移除了 `ROI enlarged` 面板与放大箭头。表格采用与 formal_ablation 相近的灰线、白底、分组结构；Optimal Strict ΔEI / Induced CQ 两张表的 `Method` 列按方法进行真正的纵向视觉合并（Single-stage / Two-stage 各占一个大格）。

## 运行方式

1. `--preflight`：只读检查来源敏感性结果、K10 及真实空间图、18 个严格重建模型是否存在。不会训练模型或重新计算 EI。
2. 正常运行：从 `--sensitivity-root` 中经科学计算完成并写入 `manifest.json` 的 EI/CQ 数据，及 `--optimal-runs-root` 中每个 strict 结果目录，自动写 CSV、PNG、PDF、审计文件与 output_manifest.json。
3. `--allow-incomplete`：仅用于图形开发/预览，保留未完成格子的 `—`，生成 PARTIAL_NO_K10/INCOMPLETE 文件，manifest 标记为 `incomplete_exploratory`。

示例（实际在沙箱通过的一次调用；请按服务器真实文件位置替换路径）：

```bash
python -m mignet_ce.downstream.formal_core_analysis \
  --sensitivity-root /mnt/data/formal_visual_review_v4/sensitivity_recomputed \
  --data-root /mnt/data/causality_sandbox \
  --optimal-runs-root /mnt/data/formal_code_stage1_work/validation_outputs \
  --output-root /mnt/data/formal_visual_review_v4/fresh_chain \
  --figure-pair '11.5->12.5' \
  --allow-incomplete
```

同样的参数加 `--preflight` 可预先审计。如果 K10 或完整 18 条 strict 结果不全，**不使用** `--allow-incomplete` 的正式运行会拒绝产出完整论文表。

### 实验边界

此模块是**Results 数据汇总 + 图表导出流水线**，**不是原始数据→全部最优粗粒化训练的一键工作流**。最优粗粒化的 Single-stage 与 Two-stage 需要先用经过验证的正式入口分别训练并保存 strict JSON、S、PIJ。切勿使用训练 ΔEI 替代 strict ΔEI。

Natural CQ 和 Optimal CQ 是 information-retention CQ，**不是直接 strict 宏观转移核的 JS 预测闭合**。低信号值保留在审计文件，不通过无标识去掉 dagger 来伪装有效。当前沙箱尚无 K10 数据，完整论文输出尚未验证。
