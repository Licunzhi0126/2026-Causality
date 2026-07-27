# Native V7 × WYT 粗粒化运行说明

## 本分支解决的问题

旧 `wyt_cg_regsim_v7` 使用 H5AD `obs` 中的 Regulon/Module 活性 `R`，并不等于原生 V7 的真实 GRN 状态 `G`。本分支新增独立入口，不改动原生 `compare_NG_kl_sinkhorn_grnanchor_v7.py`：

- spot 侧：真实 CCI → directed joint NMF 得到 `N`；真实表达 + 真实 GRN → 原生 V7 `G`。
- WYT：只学习 spot→macro 的软分配 `S`。
- 推荐训练接口 `--macro-feature-mode recompute_g`：每个 epoch 先池化真实表达，再用真实 GRN 重算 macro `G`。
- 严格评价：`S.T @ A @ S` 后重新运行 NMF 得到 macro `N`；池化表达后重算 macro `G`；再调用 native V7 KL-Sinkhorn。
- 同时导出 `hardK`、`Keff`、最大簇占比和 assignment confidence，避免名义 `K=64` 掩盖原型塌缩。

## 推荐探索配置

```bash
python -u scripts/run_native_v7_wyt_realdata.py \
  --stage-t 11.5 \
  --stage-tp 12.5 \
  --h5ad-t "/path/to/spot_heart_11.5.h5ad" \
  --h5ad-tp "/path/to/spot_heart_12.5.h5ad" \
  --cci-t "/path/to/spot_heart_11.5_CCI_total.npz" \
  --cci-tp "/path/to/spot_heart_12.5_CCI_total.npz" \
  --cci-index-t "/path/to/spot_heart_11.5_index.tsv" \
  --cci-index-tp "/path/to/spot_heart_12.5_index.tsv" \
  --grn-t "/path/to/11.5/grn_edges.csv" \
  --grn-tp "/path/to/12.5/grn_edges.csv" \
  --out-root "/path/to/output/native_v7_wyt" \
  --k 64 \
  --epochs 300 \
  --seeds 42 \
  --graph-modes cci_g_integrated \
  --local-graph-modes all_features \
  --macro-feature-mode recompute_g \
  --lambda-min-usage 100 \
  --lambda-max-usage 100 \
  --min-usage-frac 0.5 \
  --max-usage-frac 2.0 \
  --device cpu
```

## 关键输出

每个运行目录中的 `all_results.csv` 同时包含：

- `training_deltaEI`：真正用于当前训练接口的 ΔEI；
- `deltaEI_training_interface_pool_NG`：仅作 pooled 特征诊断，不应在 `recompute_g` 模式下冒充训练分数；
- `deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G`：正式推荐的严格结果；
- `deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G`：宏观 CCI 行归一化敏感性结果；
- `assignment_t` / `assignment_tp`：hardK、Keff、最大簇占比、置信度。

## 当前限制

训练时 macro `G` 已经严格重算；macro `N` 仍使用 pooled surrogate，严格事后才对 `S.T @ A @ S` 重新 NMF。当前 V7 的 N 校正权重较弱，因此训练与严格结果几乎一致；提高 N 权重前，需要进一步设计可微分 macro N 或交替优化。
