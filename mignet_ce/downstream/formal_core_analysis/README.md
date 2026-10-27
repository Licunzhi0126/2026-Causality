# Formal Core Analysis：Closure JS Direct Bits（2026-10 增量）

本模块在**不改变训练算法及已经确认的 ΔEI / Keff 三格表**的前提下，额外生成两张独立的 Closure JS Direct（bits）表。两种来源的 JS 口径相同，但输入来源不同，不与 CQ 或 Strict Reconstruction 混同。

## 1. 两张表与准确的数据口径

**自然层级** `tables/table_natural_closure_js_direct_bits.{csv,png,pdf}`：直接选择正式 `formal_sensitive_analysis` 的 `alpha_natural_cq.csv` 里已有的 `closure_JS_direct_bits`，不重新拟合。列固定为 Spot→K150、Spot→K40、K150→K40、Spot→K10*；行是 11.5→12.5、12.5→13.5、11.5→13.5。K150→K40 必须记录真实 `mapping_method=spot_overlap_soft_contingency`，不当作嵌套标签。K10 需要实际跑完 formal sensitivity 的 K10 输入及结果；不得用空白或 K40 冒充。

**最优粗粒化** `tables/table_optimal_closure_js_direct_bits.{csv,png,pdf}`：使用各个**已经完成且通过 `inspect_optimal_run` 审计**的 18 个 checkpoint：

- `PIJ_micro_train.npy`：微观训练 PIJ，形状 `N_t × N_tp`；
- `S_t.npy` / `S_tp.npy`：训练出的源/目标软分配，形状 `N_t × K_t` / `N_tp × K_tp`；
- `PIJ_macro_train.npy`：**训练**宏观 PIJ，形状 `K_t × K_tp`。

每个源节点的两组未来宏观分布分别为 `R_i=(PIJ_micro_train @ S_tp)[i]` 与 `\hat R_i=(S_t @ PIJ_macro_train)[i]`。指标是所有源节点 `JS(R_i, \hat R_i)`（对数底 2）的简单平均；**越小仅代表直接预测分布越接近，不代表独立测试集泛化或整体 CQ 更好**。JS 来自和 Sensitive Analysis **相同**的 `dynamic_closure.analysis.js_rows`。严格检查 shape、非负、有限、行随机性；不会用诱导 `Q_induced` 替代训练 PIJ。

## 2. 运行（复用正式结果，不会重训练）

在完整、真实的敏感性输出，以及两个方法 × 三尺度 × 三时间对的 18 组训练目录已生成的情况下：

```bash
python -m mignet_ce.downstream.formal_core_analysis \
  --sensitivity-root "/home/jovyan/work/2026 Causality/output/formal_sensitive_analysis" \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --optimal-runs-root "/home/jovyan/work/2026 Causality/output/formal_analysis/optimal_runs" \
  --output-root "/home/jovyan/work/2026 Causality/output/formal_analysis" \
  --stage all --skip-figures
```

> 注意：这里的 `--sensitivity-root` 是**示例**，必须指向服务器实际存在的、含 `manifest.json`、`alpha_layer_ei.csv` 与 `alpha_natural_cq.csv` 的目录。`--optimal-runs-root` 必须确实包含已训练结果。`--skip-figures` **只跳过空间 Figure 1/2**，不跳过表格 PNG/PDF。

只输出自然表可用 `--stage natural`。如果**缺少真实 K10**，仅允许添加 `--allow-incomplete`，输出 `table_natural_closure_js_direct_bits_PARTIAL_NO_K10.*`，其中 K10 的 CSV 为 NaN、PNG/PDF 为 `—`，并明确标记为探索性，禁止称为完整论文结果。

检查来源和完整性（不生成结果）：增加 `--preflight`。查证完成请读取 `output_manifest.json` 与 `audit/trained_optimal_run_inventory.csv`；在此轮新增的 inventory 中可核查 `closure_JS_direct_bits` 列及对应 run_dir。

## 3. 输出及验收

- 自然和最优两个表各三份 CSV / PNG / PDF；排版为用户已确认的结构，最佳方法的 Method 列视觉纵向合并；仍保留原先三格 ΔEI / Keff 表。
- 每个正式最优数值须回溯到 `run_dir` 下的四个真实 checkpoint 数组；无数值伪造，不做 checkpoint 的再次择优。
- 如果数据未齐或某个训练记录不合格，正式 `--stage all` 会失败，不能标记 complete；`--allow-incomplete` 不生成缺数据的正式最优 JS 表。
- 本增量包含新测试文件，可执行 `python -m pytest -q mignet_ce/downstream/formal_core_analysis/tests/test_closure_js.py`。

## 4. 关于真实数据验证

`CCI_total.npz` 属于**真实的 CCI 网络，不等价于正式训练的 `PIJ_micro_train.npy`**。用真实 CCI 原始网络验证矩阵顺序、软映射及 JS 数值恒等式，只是**真实输入驱动的数学一致性 smoke test**。要给出正式的 3×4 自然和 6×3 最优科学结果，依然需要实际 formal sensitivity 输出，以及 18 个正式 checkpoint。不要用 CCI 归一化的试验矩阵去填正式表格。
