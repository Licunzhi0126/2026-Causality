# 空间 domain 图

从仓库根目录运行。默认时间对是小鼠心脏 E12.5 → E13.5。可以用 `--time-pair 11.5 12.5` 或 `--time-pair 11.5 13.5` 绘制其他正式时间对。

## 本地预览

```bash
python -B -m mignet_ce.downstream.domain_visualization --preview
```

默认输出到 `output/domain_visualization/heart/12.5_to_13.5/`。四张图分别为 `natural_domains`、`optimal_domains_spot`、`optimal_domains_seurat_k150`、`optimal_domains_seurat_k40`，每张都导出 PNG 和 PDF。自然图和对照图的输入行使用真实坐标及 Seurat 标签。对照图的粗粒化面板是占位图，K_eff 标为不可用。`--natural-only` 可单独导出自然图。重复运行时可指定新的 `--output-root`。

## 正式结果

```bash
python -B -m mignet_ce.downstream.domain_visualization \
  --data-root /path/to/E1S1_domain_factory \
  --optimal-runs-root /path/to/formal_analysis/optimal_runs \
  --output-root output/domain_visualization_formal
```

每次读取所选时间对的 12 组正式结果，目录结构为 `<method>/<input_scale>/K<k>/<source>_to_<target>/`。三种输入尺度分别使用训练设定 K=40、40、10。程序复用 Formal Core 审计，验证方法、输入尺度、时间对、软分配和 K_eff，再核对训练 ID 与输入 H5AD。四列从左到右为 two-stage、single-stage、two-stage_noKeff、two-stage_noKeff_HardEI。

每格内左图为源阶段，右图为目标阶段。所有面板都在同一阶段的原始 Spot 坐标上绘制。Spot 输入用灰色点位，Seurat 输入用原始 domain 标签。粗粒化颜色取最大分配概率对应的 domain。对 Seurat 输入，每个 Spot 继承其所属 Seurat domain 的粗粒化标签。颜色用于区分各面板内的类别。K_eff 由训练输入单元上的完整软分配核算，源、目标阶段各标一个值。
