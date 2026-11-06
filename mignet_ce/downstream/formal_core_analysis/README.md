# Formal Core Analysis：五方法与 Closure JS Direct Bits（2026-11 增量）

本模块保留原 single-stage / two-stage 方法，增加三个项目内实现的对照方法，统一生成 ΔEI / Keff、CQ 和 Closure JS Direct（bits）表。两种来源的 JS 口径相同，但输入来源不同，不与 CQ 或 Strict Reconstruction 混同。

## 1. 两张表与准确的数据口径

**自然层级** `tables/table_natural_closure_js_direct_bits.{csv,png,pdf}`：直接选择正式 `formal_sensitive_analysis` 的 `alpha_natural_cq.csv` 里已有的 `closure_JS_direct_bits`，不重新拟合。列固定为 Spot→K150、Spot→K40、K150→K40、Spot→K10*；行是 11.5→12.5、12.5→13.5、11.5→13.5。K150→K40 必须记录真实 `mapping_method=spot_overlap_soft_contingency`，不当作嵌套标签。K10 需要实际跑完 formal sensitivity 的 K10 输入及结果；不得用空白或 K40 冒充。

**最优粗粒化** `tables/table_optimal_closure_js_direct_bits.{csv,png,pdf}`：使用各个**已经完成且通过 `inspect_optimal_run` 审计**的 45 个 checkpoint：

- `PIJ_micro_train.npy`：微观训练 PIJ，形状 `N_t × N_tp`；
- `S_t.npy` / `S_tp.npy`：训练出的源/目标软分配，形状 `N_t × K_t` / `N_tp × K_tp`；
- `PIJ_macro_train.npy`：**训练**宏观 PIJ，形状 `K_t × K_tp`。

每个源节点的两组未来宏观分布分别为 `R_i=(PIJ_micro_train @ S_tp)[i]` 与 `\hat R_i=(S_t @ PIJ_macro_train)[i]`。指标是所有源节点 `JS(R_i, \hat R_i)`（对数底 2）的简单平均；**越小仅代表直接预测分布越接近，不代表独立测试集泛化或整体 CQ 更好**。JS 来自和 Sensitive Analysis **相同**的 `dynamic_closure.analysis.js_rows`。严格检查 shape、非负、有限、行随机性；不会用诱导 `Q_induced` 替代训练 PIJ。

## 2. 运行（复用正式结果，不会重训练）

在完整、真实的敏感性输出，以及五个方法 × 三尺度 × 三时间对的 45 组训练目录已生成的情况下：

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
- 对应测试位于 `tests/test_wyt_two_stage.py`、`tests/test_formal_core_analysis.py`、`tests/test_multiscale_maturity_orchestrator.py` 和 `tests/test_generalization_audit.py`。

## 4. 关于真实数据验证

`CCI_total.npz` 属于**真实的 CCI 网络，不等价于正式训练的 `PIJ_micro_train.npy`**。用真实 CCI 原始网络验证矩阵顺序、软映射及 JS 数值恒等式，只是**真实输入驱动的数学一致性 smoke test**。要给出正式的 3×4 自然和 15×3 最优科学结果，依然需要实际 formal sensitivity 输出，以及 45 个正式 checkpoint。不要用 CCI 归一化的试验矩阵去填正式表格。


## 5. 最终方法与 single-stage 对照

最终方法是 `maturity_cci_grn_two_stage`；`complete_combined_coarse_maturity_cci_grn` 保留为 single-stage 对照。
Formal Core 固定顺序为 Single-stage、Two-stage、two_stage_noKeff、two_stage_noKeff_HardEI、single_stage_cq_only。每个方法覆盖 Spot/K150/K40 与三个时间对，各 9 组，共 45 组；三张最优表各 15 行。
所有方法继续使用 GRN=`legacy_nmf`、CCI=`laplacian_hks`、alpha=0.20。

多尺度入口 `scripts/run_multiscale_maturity_cci_grn.py` 默认运行 two-stage；需要对照时显式选择 single-stage。
单次入口 `scripts/run_wyt_deltaei_coarse_grain.py` 仍要求显式提供 `--method`。
泛化入口支持 `--method two_stage` 或完整方法名，结果保存在 `two_stage/` 下。
历史退役方法的训练结果保留在原处，Formal 收集器会排除它们；旧方法名不能用于启动训练。

Two-stage 继续复用 `complete_combined_coarse_maturity_cci_grn` 输入前端。
本次退役清理不改变 two-stage 模型、损失公式、训练循环、checkpoint 规则及公共 CCI/GRN/PIJ 数学。
保留 `--lambda-retain-norm`、`--lambda-retain-floor` 等 two-stage 参数。

## 6. 完整流程与续跑

`python -m mignet_ce.downstream.formal_core_analysis.run_pipeline` 负责输入检查、五个方法的训练与 Formal 输出。
必填参数为 `--data-root`、`--grn-root`、`--cci-root`、`--developmental-root`、`--sensitivity-root` 和 `--output-root`。
先添加 `--dry-run` 检查输入并打印命令；正式运行时移除该参数。

- `--resume` 复用已有有效结果，并核对方法身份。
- `--resume --resume-incomplete` 只对缺失或无效结果重新运行；原失败产物备份在 `optimal_runs` 之外，保留已有备份。
- 不注入其他方法的训练或 checkpoint 参数，two-stage 使用自身配置。

布局预览使用 `python -m mignet_ce.downstream.formal_core_analysis.preview`，默认输出到 `output/formal_core_analysis_preview/`。
预览只有占位符，不训练模型、不产生科学测量 CSV；已有非空目录不会被覆盖。

## 7. 回归验证

在具备 PyTorch、NumPy、SciPy、pandas、h5py、matplotlib 和 pytest 的环境中运行：

```bash
python -B -m pytest -q tests/test_wyt_coarse_frontends.py tests/test_wyt_two_stage.py tests/test_wyt_cq_only.py tests/test_multiscale_maturity_orchestrator.py tests/test_formal_core_analysis.py tests/test_generalization_audit.py
```

测试覆盖旧方法和专属参数被拒绝、two-stage 训练和 checkpoint 输出、真实文件格式前端与宏观 PIJ 梯度、多尺度默认方法与目录保护、45 组结果审计及泛化流程。
合成夹具不代表正式生物实验，不得用于填写论文结果。每次指定 pytest `--basetemp` 时应使用新的临时目录，以保留已有测试产物。


## 8. 三个新增方法及配置契约

所有实现都位于本项目；reference/压缩包仅用于开发参考，运行时不读取、导入或解压它们。

| 方法 ID（同时为结果目录名） | 训练协议 | EI 保留比例 | Keff / dead-usage |
|---|---|---|---|
| two_stage_noKeff | 先 ΔEI，后闭合与信息保留 | 0.85 | 两阶段均关闭惩罚及 Keff 下限 |
| two_stage_noKeff_HardEI | 同上 | 1.0 | 同上 |
| single_stage_cq_only | 全部 epochs 优化 CQ | 无 ΔEI 目标或保底 | 关闭惩罚及 Keff 下限 |

HardEI 是方法名称：训练仍使用平方惩罚，严格选模要求达到对应 EI 比例；没有严格合格结果时保留原生放宽回退。因此最终 EI 不保证达到第一阶段参考值的 100%。summary.json 与 Formal inventory 记录严格条件和回退情况。

CQ-only 保留参考实现的对齐、方差、局部平滑、锐化、发育、原型分离、闭合、组内/组间动力学及有效信息量要求。按有效信号候选中的最大 CQ 保存 best_cq.pt，同时导出 best_model.pt；单阶段不创建 best_joint.pt。DeltaEI 可以为负。CQ 训练目标使用带信号分母下限的 cq_safe，结果表始终从持久化 PIJ/S 重算原始 induced CQ。

三个方法共享原 maturity+CCI+GRN 输入和模型。training_config.py 同时为 CLI 与泛化审计构建真实配置，防止 0.85/1.0 或无 Keff 下限的结果相互误用。原方法的历史 config.json 缺少 keff_policy 时，按 legacy_floor 兼容。

泛化使用 scripts/run_generalization_audit.py --task spot40 --method 后接上述三个方法名之一，分别生成器官为行、时间对为列的 ΔEI、CQ 两张 CSV/PNG/PDF 和组合图。支持 heart/lung/brain，默认只选 lung/brain；使用 --organs heart lung brain 可选全三器官。每个方法的数据隔离在 output-root/<method>/，auto 会核验并复用已有匹配结果，只补跑缺失单元。

完整 Formal 示例（服务器 Bash，不定义路径变量）：

```bash
cd "/home/jovyan/work/2026 Causality"
python -m mignet_ce.downstream.formal_core_analysis.run_pipeline \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --sensitivity-root "/home/jovyan/work/2026 Causality/output/formal_sensitive_analysis" \
  --output-root "/home/jovyan/work/2026 Causality/output/formal_analysis" \
  --maturity-mode existing --epochs 1500 --device cuda --resume --resume-incomplete
```

上述 sensitivity-root 和 developmental-root 必须指向服务器真实已有结果；完整预检要求自然层级敏感性结果已覆盖 K10 和 K150→K40 direct JS。运行时先将同一条命令加 --dry-run 可检查输入且不写文件。
