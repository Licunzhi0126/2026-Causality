# 因果涌现下游分析基础设施

## 正式四映射全量入口（full-v2，2026-08-13）

正式下游分析使用现有主题目录，不使用单体 `unified_suite.py`：

- `determinism_degeneracy/`：四映射 EI、ΔEI 与分解；
- `spatial/`：state EI 空间图、有效状态数与空间形态；
- `dynamic_closure/`：信息闭合恒等式、三面板闭合主图和六组表示一致性；
- `null_model/`：matched partition null；
- `grn_cci/`：共同 spot 基底上的 GRN/CCI mechanism；
- `fate_path/`：四时点 macro fate chain；
- `perturbation/`：targeted 与 matched-random 虚拟扰动。

四个正式宏观研究对象为：

1. Seurat K150；
2. Seurat K40；
3. `complete_combined_coarse`；
4. `complete_combined_coarse_maturity_cci_grn`。

Spot 仅作为微观 EI/空间参考，不作为第五个粗粒化结果。

```powershell
python -u scripts/run_unified_downstream_analysis.py `
  --data-root <E1S1_domain_factory> `
  --cache-root <full_cache_root> `
  --output-dir <new_output_directory> `
  --device auto
```

正式入口锁定完整 benchmark：两个 optimized 方法乘三个相邻时间对，共六个
DeltaEI 作业；`K=40`、`epochs=1500`、所有 pair 的 NMF 均为 `5/300`、matched
null `200`、perturbation random `200`。正式 CLI 不提供降低这些参数的选项，
也不会读取 preview、残缺或 manifest 不匹配的缓存。正式缓存位于
`<cache-root>/full/<profile-id>/`，旧结果不会被删除或覆盖。

`full-v2` 严格区分两种状态空间：普通 EI、determinism/degeneracy、fate、
mechanism 与 perturbation 使用完整 model Q 和完整 soft K-state assignment；
只有动力学闭合、NMI/ARI、matched-partition null 与空间形态诊断可以在模块内部
使用 hard partition。optimized 的正式 ΔEI 必须由 `PIJ_macro_train.npy` 与其匹配的
`PIJ_micro_train.npy` 重算，并以 `1e-5` 容差匹配训练 `summary.json`，否则正式流程
直接失败。

新的正式缓存协议为 `full_model_space_v2`，profile ID 类似：

```text
fullv2_k40_e1500_nmf5_i300_seed20260809
```

旧 `..._l60_...` 缓存不会被删除，但不会被 full-v2 静默复用。

正式输出为 11 张 CSV 表、9 张 PNG、9 张 PDF，以及
`audit/validation_checks.csv`、`audit/deltaei_contract.csv` 和 `audit/manifest.json`。

独立复核命令：

```bash
python -u scripts/audit_unified_deltaei_contract.py \
  --full-cache-root /path/to/cache/full/fullv2_profile_id \
  --metrics-csv /path/to/results/tables/01_metrics.csv
```

下文是保留的历史 K150/K40 分析入口与兼容说明。

本目录只提供分析、数据读取和六面板绘图函数，不包含可直接运行的模块入口。
正式代码不会读取或导入 `output/report` 中的博物馆代码。

分析与绘图实现按图的研究主题组织：

- `determinism_degeneracy/`：图 1
- `spatial/`：图 2、图 7
- `dynamic_closure/`：图 3、图 4、图 6
- `null_model/`：图 5
- `grn_cci/`：图 8
- `fate_path/`：图 9
- `perturbation/`：图 10

顶层 `analysis.py` 和 `plots.py` 保留原有导入路径，仅作为兼容入口。

以下是已弃用的历史 K150/K40 兼容入口，不是正式四表示 benchmark：

```powershell
python scripts/run_downstream_analysis.py analyze `
  --data-root data/mouse_embyro/E1S1_domain_factory `
  --metrics-csv output/pij_export_run/metrics.csv `
  --pair-archive data/mouse_embyro/E1S1_domain_factory/pij/network=light_cci_grn/pij=NG_KLot/organ=heart/pair=seurat_k150_to_seurat_k40 `
  --output-dir output/downstream_six_panel
```

已有 `tables/*.csv` 时只重画：

```powershell
python scripts/run_downstream_analysis.py render `
  --results-dir output/downstream_six_panel `
  --data-root data/mouse_embyro/E1S1_domain_factory
```

输出包括 17 张审计表、十张高分辨率 PNG、十张矢量 PDF、`findings.json`
和 `manifest.json`。随机粗粒化、GRN/CCI 关联和 Pij 行同质化扰动的解释边界会写入
`findings.json`。
