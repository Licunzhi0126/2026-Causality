# Formal Core Analysis：Closure JS Direct Bits（2026-10 增量）

本模块在**不改变训练算法及已经确认的 ΔEI / Keff 三格表**的前提下，额外生成两张独立的 Closure JS Direct（bits）表。两种来源的 JS 口径相同，但输入来源不同，不与 CQ 或 Strict Reconstruction 混同。

## 1. 两张表与准确的数据口径

**自然层级** `tables/table_natural_closure_js_direct_bits.{csv,png,pdf}`：直接选择正式 `formal_sensitive_analysis` 的 `alpha_natural_cq.csv` 里已有的 `closure_JS_direct_bits`，不重新拟合。列固定为 Spot→K150、Spot→K40、K150→K40、Spot→K10*；行是 11.5→12.5、12.5→13.5、11.5→13.5。K150→K40 必须记录真实 `mapping_method=spot_overlap_soft_contingency`，不当作嵌套标签。K10 需要实际跑完 formal sensitivity 的 K10 输入及结果；不得用空白或 K40 冒充。

**最优粗粒化** `tables/table_optimal_closure_js_direct_bits.{csv,png,pdf}`：使用各个**已经完成且通过 `inspect_optimal_run` 审计**的 36 个 checkpoint：

- `PIJ_micro_train.npy`：微观训练 PIJ，形状 `N_t × N_tp`；
- `S_t.npy` / `S_tp.npy`：训练出的源/目标软分配，形状 `N_t × K_t` / `N_tp × K_tp`；
- `PIJ_macro_train.npy`：**训练**宏观 PIJ，形状 `K_t × K_tp`。

每个源节点的两组未来宏观分布分别为 `R_i=(PIJ_micro_train @ S_tp)[i]` 与 `\hat R_i=(S_t @ PIJ_macro_train)[i]`。指标是所有源节点 `JS(R_i, \hat R_i)`（对数底 2）的简单平均；**越小仅代表直接预测分布越接近，不代表独立测试集泛化或整体 CQ 更好**。JS 来自和 Sensitive Analysis **相同**的 `dynamic_closure.analysis.js_rows`。严格检查 shape、非负、有限、行随机性；不会用诱导 `Q_induced` 替代训练 PIJ。

## 2. 运行（复用正式结果，不会重训练）

在完整、真实的敏感性输出，以及四个方法 × 三尺度 × 三时间对的 36 组训练目录已生成的情况下：

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
- SHR 接入测试位于 `tests/test_shr_method_8.py`、`tests/test_shr_method_10.py`、`tests/test_shr_integration.py` 和 `tests/test_formal_core_shr.py`。

## 4. 关于真实数据验证

`CCI_total.npz` 属于**真实的 CCI 网络，不等价于正式训练的 `PIJ_micro_train.npy`**。用真实 CCI 原始网络验证矩阵顺序、软映射及 JS 数值恒等式，只是**真实输入驱动的数学一致性 smoke test**。要给出正式的 3×4 自然和 12×3 最优科学结果，依然需要实际 formal sensitivity 输出，以及 36 个正式 checkpoint。不要用 CCI 归一化的试验矩阵去填正式表格。


## 5. SHR Method 8 / SHR Method 10

正式最优粗粒化表的方法顺序固定为：

| 表格显示名称 | CLI 方法名 | 核心实现 |
|---|---|---|
| Single-stage | `complete_combined_coarse_maturity_cci_grn` | 原有单阶段方法 |
| Two-stage | `maturity_cci_grn_two_stage` | 原有两阶段方法 |
| SHR Method 8 | `shr_method_8` | 联合时间点标准化/PCA32、KNN 图、线性分配头 |
| SHR Method 10 | `shr_method_10` | CCI 自注意力、双向跨时间注意力、门控融合、余弦学习率 |

两种 SHR 方法的前端复用已有 maturity+CCI+GRN 准备流程。Formal Core 继续使用 GRN=`legacy_nmf`、CCI=`laplacian_hks`、alpha=0.20。每个方法在 Spot/K150/K40 与三个时间对上各有 9 组，合计 36 组。三张最优表各 12 行；每个方法的 Method 单元格跨三行合并。自然层级表与 Figure 1/2 算法未改动。

`run_pipeline` 默认调度全部四个方法。沿用原来的命令与参数即可；已有合格结果可使用 `--resume` 复用。正式结果完整性审计现在要求四个方法全部就绪，旧的 18 组结果仍可读取，但不能单独标记为四方法完整结果。

单方法入口仍是 `scripts/run_wyt_deltaei_coarse_grain.py --method shr_method_8` 或 `--method shr_method_10`，所需的 H5AD、CCI/index、GRN、maturity、K 与输出目录参数沿用原入口。Method 8 默认 `--graph-projection-dim 32`，不足 32 个联合样本或输入特征时明确报错，不自动改维度；小输入测试可显式设置较低维度。Method 10 默认 `--attention-heads 4`，要求 `hidden_dim` 能被 head 数整除、两层注意力。新增 `--lambda-retain`，原有方法的默认参数与训练循环不变。

SHR 最佳 checkpoint 保存在 `best_model.pt`，仍按原实现要求有限指标、正 DeltaEI 以及双端 Keff 下限；按 DeltaEI、CQ、retained information 依次择优。无合格 checkpoint 时失败，不把诊断模型当作正式结果。输出包括原项目的 `S_t/S_tp`、训练 micro/macro PIJ、assignments 和 manifests，直接兼容 `inspect_optimal_run`。Method 8 另保存 `graph_projection.npz`。

### 仅预览排版（不跑实验）

```bash
python -m mignet_ce.downstream.formal_core_analysis.preview \
  --output-root output/formal_core_analysis_shr_preview
```

专用 preview 入口使用与正式表相同的绘图布局，仅填文本占位符 `—`，生成三张 PNG/PDF 和 `preview_manifest.json`，明确标记 `PREVIEW`、`models_trained=false`、`contains_experimental_measurements=false`。不读取科学结果，不创建结果 CSV/NPY；目标目录非空时拒绝覆盖。正式 renderer 的有限数值检查保持有效。

### 实现来源与测试边界

源码从提供的 SHR.zip 中迁入以下本地模块；运行和测试都不导入或执行压缩包内的 Python：

| 原源码（Attention/wyt_deltaei_coarse_grain/） | 本项目模块（wyt_deltaei_coarse_grain/） |
|---|---|
| method8_model.py / method8_helpers.py / method8_training_core.py | shr_method_8_model.py / shr_method_8_helpers.py / shr_method_8_training_core.py |
| pca32_knn_linear_head_single_stage_trainer.py | shr_method_8_trainer.py |
| bidirectional_cross_time_attention_model.py | shr_method_10_model.py |
| bidirectional_cross_time_attention_single_stage_trainer.py | shr_method_10_trainer.py |
| coarse_objectives.py / coarse_training_utils.py / graph_utils.py | shr_objectives.py / shr_training_utils.py / shr_graph_utils.py |

保留模型结构、损失公式、checkpoint 规则和默认训练设置；适配本项目方法身份、导入路径、CLI 和保存接口。额外固定 PCA 的 random_state 为配置 seed（包括随机化 SVD 分支），并补充 Method 10 的参数校验。两种新方法不依赖 Two-stage 的模型或训练循环。

测试使用小型合成矩阵、临时 H5AD/CCI/GRN/maturity 文件与短 CPU 训练，覆盖梯度、模型恢复、checkpoint 矩阵审计、四方法命令与表格完整性、预览及已有目录保护。测试夹具不代表正式生物实验，不得用来填正式论文表格。输出位于独立的 `output/shr_method_tests/`；不启动 36 组正式训练。

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=output/shr_method_tests/NEW_EMPTY_RUN_ID \
  tests/test_shr_method_8.py tests/test_shr_method_10.py \
  tests/test_shr_integration.py tests/test_formal_core_shr.py
```

测试前先创建 `output/shr_method_tests/`。每次使用新的 basetemp，以保留此前测试产物。

迁移源压缩包 SHA256：`23a9d5d4780b1c6648d1128ef9e6726890c5a68236d59310a6c4bea9382cb780`。
