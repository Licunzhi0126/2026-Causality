# Native V7 × WYT 适配变更

- 新增 `mignet_ce/pij/compare/native_v7_torch.py`：可微分 native V7 N+G 成本和 Sinkhorn。
- 新增 `scripts/run_native_v7_wyt_realdata.py`：真实 H5AD/CCI/GRN 单实验入口。
- 新增 `scripts/run_native_v7_wyt_realdata_parallel.py`：独立进程并行入口。
- 新增 `scripts/discover_native_v7_real_inputs.py`：输入发现和形状/索引审计。
- 更新 `scripts/run_native_v7_wyt_study.py`：directed joint NMF 使用数学等价的稀疏精确乘法更新，支持近稠密大 CCI。
- 更新 `wyt_deltaei_coarse_grain/trainer.py`：支持 `legacy_features`、`coords`、`all_features` 局部图；增加用量约束参数；根据矩阵密度选择稀疏/稠密 tensor。
- 更新 H5AD 读取：在缺少 anndata 的环境中直接用 h5py 读取 CSC/CSR、obs、obsm/spatial 和 layer。
- 新增 `--macro-feature-mode recompute_g`：池化表达后通过真实 GRN 每轮重算 macro G。

原 `compare_NG_kl_sinkhorn_grnanchor_v7.py` 与旧 `wyt_cg_regsim_v7` 均保留，不覆盖旧实验入口。
