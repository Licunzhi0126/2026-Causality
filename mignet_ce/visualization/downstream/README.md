# 因果涌现下游分析基础设施

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

唯一命令行入口：

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
