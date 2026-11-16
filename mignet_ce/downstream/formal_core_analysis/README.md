# Formal Core Analysis：当前正式流程（2026-11-16）

当前最优粗粒化流程仅运行 `formal_two_stage`，并严格核验协议 `formal_two_stage_main`。
默认器官顺序为 Lung / Brain / Heart；每个器官覆盖 Spot、K150、K40 三种输入以及三个时间对，合计 27 组。
K 配置沿用 Spot→40、K150→40、K40→10。`--organs heart` 可单独选择器官。
旧方法标签、兼容函数和历史输出保留，但不会进入当前最优表或被用于当前训练。

## 默认 KNN 构图：all_features

`formal_two_stage` 的 Python 配置和原生 CLI 默认使用 `local_graph_mode=all_features`；Formal Core 流水线也会显式传入这个模式。
原有其他方法保留 `legacy_features` 默认值。显式传入 `--local-graph-mode legacy_features` 或 `coords` 仍然生效。
`--knn-k` 默认 30；`--local-dims` 默认 2，仅在 `legacy_features` 模式下影响邻居选择。

当前正式前端的编码器输入为按列标准化的 `[N_CCI, G_GRN]`：CCI Laplacian-HKS 默认 5 维，
GRN legacy_nmf 默认输出 64 维，共 69 维。两端特征合并计算列统计量后再拆回各时间点。
`all_features` 使用这 69 维的完整欧氏距离，在各时间点内部独立选择近邻；随后对称化、加自环、按行归一化，供 GNN 聚合使用。
构图在训练开始时固定，不根据每个 epoch 的隐藏表示重建；`lambda_local=0` 仍会通过 GNN 使用这张图。
空间坐标对应单独的 `coords` 模式，发育伪时间通过发育损失使用。
这次仅调整默认配置和核验，不改变原有 all_features 算法或 CCI/GRN 特征提取。

注意：按列标准化不等于 CCI/GRN 两组特征等权。当前欧氏距离会累加所有维度；
`alpha_cci=.20` 用于 PIJ 计算，没有作为 KNN 的组别权重。

训练、收集和只读 preflight 都支持 `--local-graph-mode`、`--knn-k`、`--local-dims`，应使用一致设置。
完整正式结果的 `config.json` 必须明确保存这三个字段；审计 inventory 和 output manifest 会记录它们。
默认收集仅接收 all_features / KNN=30，其他图配置须显式指定；不同有效构图设置不会混入同一张表。
`all_features` 与 `coords` 忽略 local_dims，历史审计仍保留保存的原值。

续跑发现旧结果的构图模式或有效 KNN 参数与请求不符，会在备份、覆盖或启动训练之前停止。
请保留旧结果并指定新的 output-root 重新训练。Stage 1 身份签名包含构图配置，旧图的 Stage 1 也不能作为新图的共同起点。
不会因为默认值改变就把旧结果的缺失字段补成 all_features。

## 当前两张最优表

每张表有器官×输入尺度 9 行，三个时间对分组列；输出 CSV、PNG、PDF：

- `tables/table_optimal_deltaei_cq.*`：每个时间对下为训练 checkpoint 的 ΔEI（bits）、原始 induced CQ。
- `tables/table_optimal_js_hardk_keffhard.*`：每个时间对下为 direct JS（bits）、源时间点 Gap、目标时间点 Gap。
- Gap = HardK − KeffHard。HardK 为 argmax 硬分配的非空簇数；KeffHard 为其成员计数占比的熵指数。两端分别从持久化 S 矩阵核验，不使用 soft Keff 替代。
- JS 使用保存的训练微观 PIJ、两端 S 和训练宏观 PIJ；CQ 使用原有 induced 信息预算口径。
- 方法/器官/协议不符、缺失、重复或非有限值均拒绝作为完整表输出；不会择优挑选 seed。
- 审计文件 `audit/trained_optimal_run_inventory.csv` 包含器官、协议、两端 HardK、KeffHard 和 Gap。

仅从正式训练结果生成最优表，不依赖自然 sensitivity：

```bash
python -m mignet_ce.downstream.formal_core_analysis --stage optimal --data-root /path/to/E1S1_domain_factory --optimal-runs-root /path/to/formal_analysis/optimal_runs --output-root /path/to/formal_analysis --skip-figures
```

训练全部三器官并汇总最优表：

```bash
python -m mignet_ce.downstream.formal_core_analysis.run_pipeline --stage optimal --data-root /path/to/E1S1_domain_factory --grn-root /path/to/grn --cci-root /path/to/cci --developmental-root /path/to/developmental_features --output-root /path/to/formal_analysis --maturity-mode existing --epochs 1500 --device cuda --resume
```

先加 `--dry-run` 可只读检查并打印训练命令。不同器官结果隔离在
`optimal_runs/<organ>/formal_two_stage/<scale>/K<k>/<source>_to_<target>/`。
GRN/CCI/发育输入仍使用项目已有的按器官命名规则。
`--stage all` 从原始输入直接计算固定 alpha=.20 的心脏自然指标，并生成自然层级表与 Figure 1；不需要 sensitivity 目录。
Figure 2 明确选择 `--figure-organ heart --figure-input-scale spot --figure-pair '12.5->13.5'`，不按方法名排序选结果。

## 图形与预览

括弧为连续圆滑曲线；Spot→K40、K150→K10、Spot→K10 使用独立横向位置。
每个低层级 spot 均连接到其所属高层级分组的空间质心；高层级保留淡色组织轮廓并突出质心。
自然层级按 spot_id 对齐各尺度标签，不假定分组严格嵌套。Figure 2 使用训练硬分配展开到 spot 后计算空间质心。
空间质心只用于绘图；指标按真实训练输入节点计算，不能用展开后的 spot 数量重算占用指标。

```bash
python -m mignet_ce.downstream.formal_core_analysis.preview --output-root /path/to/new_empty_preview --data-root /path/to/E1S1_domain_factory
```

省略 `--data-root` 仅预览两张表；表格以 `—` 占位，不产生科学 CSV。
可附加 `--optimal-run /path/to/run --optimal-scale seurat_k150 --optimal-pair '11.5->12.5'` 预览指定 checkpoint。
旧 checkpoint 会明确标为历史方法，不会作为正式表输入。已有非空预览目录拒绝覆盖。
所有历史结果保留；收集器不执行自动清理。

---

以下为历史五方法版本说明，命令、输出名称和条件数量以以上当前流程为准。

# Formal Core Analysis：五方法与 Closure JS Direct Bits（2026-11 增量）

本模块保留原 single-stage / two-stage 方法，增加三个项目内实现的对照方法，统一生成 ΔEI / Keff、CQ 和 Closure JS Direct（bits）表。两种来源的 JS 口径相同，但输入来源不同，不与 CQ 或 Strict Reconstruction 混同。

## 独立完整流程：固定 alpha=.20，不需要 sensitivity

`run_pipeline --stage all` 的当前顺序为：只读输入预检 → 可选原生发育特征生成 → 心脏自然指标计算 → 三器官 27 组 formal_two_stage 训练 → 汇总五表两图。
不启动 alpha 扫描，不读取 sensitivity 结果目录。旧 `--sensitivity-root` 参数在完整流水线中仅为命令兼容保留，不参与计算。
单独收集器仍允许显式指定该参数读取历史自然结果；当前默认流程不走历史分支。

自然指标由 `natural_analysis.py` 负责，直接读取心脏三个时间点的 Spot/K150/K40/K10 H5AD、CCI NPZ、CCI index TSV、GRN edges 和三个 domain 层的 spot-domain map。
索引顺序必须匹配 H5AD；domain map 必须覆盖同一组 spot 和 domain。完整图表还要求图示时间点的 map 包含有限 x/y 坐标。
真实 K10 的原始数据仍然必需；缺失时在训练之前停止，不填补或推算 K10。
自然部分不使用 K10 发育特征 CSV。

沿用原自然报告的 GRN=legacy_nmf、CCI=laplacian_hks、KL+OT、alpha=.20、beta=.05、tau=.8、HKS 5 维、投影 64 维/seed=20260713、GRN top-k=50。
HKS 在每层三个时间点联合标准化；GRN 表示按时间对拟合。K150→K40 使用 spot 共归属的行归一化软映射。
CQ 采用现有 induced 信息预算；direct JS 比较 P_lower @ S_tp 与 S_t @ P_upper_direct，按源单元均匀取平均，单位 bits。
自然 PIJ 沿用原自然计算口径；最优结果沿用保存的训练 PIJ，两种来源均明确记录。

当前完整输出：

| 输出 | 内容 |
|---|---|
| `natural_analysis/natural_layer_ei.csv` | 心脏四层 × 三时间对的 12 行 EI |
| `natural_analysis/natural_hierarchy_metrics.csv` | 四种自然映射 × 三时间对的 CQ、JS 等测量明细 |
| `natural_analysis/manifest.json` | 固定协议、配置、原始输入路径/SHA256、结果 CSV SHA256 |
| `tables/table_ei_hierarchy.*` | 心脏三时间对的六种自然层级 Delta EI |
| `tables/table_cq_hierarchy.*` | 心脏 Spot→K150/K40/K10 的 CQ |
| `tables/table_natural_closure_js_direct_bits.*` | 心脏 Spot→K150/K40/K10、K150→K40 的 direct JS |
| `tables/table_optimal_deltaei_cq.*` | 肺/脑/心 × 三输入尺度，三个时间对的 Delta EI 与 CQ |
| `tables/table_optimal_js_hardk_keffhard.*` | 同上，JS 和源/目标时间点的 HardK−KeffHard |
| `figures/figure1_natural.*` | 心脏自然四层，默认 E12.5→E13.5 |
| `figures/figure2_optimal_NO_ROI_ENLARGE.*` | 默认心脏 Spot 输入、E12.5→E13.5 的正式训练结果 |

表格的 `.*` 为 CSV/PNG/PDF，空间图为 PNG/PDF。另有 `optimal_runs/` 训练产物、`audit/` 收集审计、`formal_full_pipeline_execution.json` 和 `output_manifest.json`。

服务器 Bash 完整命令（先完成现有 developmental_features 中缺失的肺/脑 K150/K40 CSV）：

```bash
cd "/home/jovyan/work/2026 Causality"
python -u -m mignet_ce.downstream.formal_core_analysis.run_pipeline \
  --stage all \
  --organs lung brain heart \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --output-root "output/formal_analysis_all_features_v1" \
  --maturity-mode existing \
  --local-graph-mode all_features --knn-k 30 \
  --alpha 0.20 --figure-pair "12.5->13.5" \
  --epochs 1500 --seed 42 --device cuda \
  --resume
```

先在同一命令末尾追加 `--dry-run`，可只读检查原始输入、K10、发育 CSV、已有自然缓存及打印全部命令，不训练、不产生输出文件。
输入齐全不等于自然指标已经计算；预检分别报告 `natural_inputs.ready` 和 `natural_k150_k40_js_ready`。
`--skip-render` 只训练；`--stage optimal` 只训练/汇总最优部分，不要求自然/K10 输入。

`--resume` 只复用配置、输入哈希和结果哈希均匹配的完整自然结果。数据变化、结果被修改或缓存不完整时停止；使用新 output-root 保留旧结果。
训练部分继续使用已有逐条件 resume / resume-incomplete 规则。`--resume-incomplete` 不覆盖不合格的自然缓存。

单独计算自然测量：

```bash
python -u -m mignet_ce.downstream.formal_core_analysis.natural_analysis \
  --data-root /path/to/E1S1_domain_factory \
  --grn-root /path/to/grn --cci-root /path/to/cci \
  --output-root /path/to/formal_analysis/natural_analysis --resume
```

单独收集时可用 `--natural-root` 指向上述测量目录；不提供时从原始输入计算到 output-root/natural_analysis，可配合 `--resume` 复用。
原始输入在收集时仍用于来源哈希核验。独立收集器的 `--preflight` 区分“输入就绪可计算”和“测量已齐可形成完整结果”。

验证：

```bash
python -B -m pytest -q tests/test_formal_core_analysis.py tests/test_multiscale_maturity_orchestrator.py tests/test_formal_two_stage_loss_ablation.py -p no:cacheprovider
```

测试使用显式合成夹具验证固定 alpha 数值与共享数学的一致性、K10/索引/映射预检、缓存防串用和五表两图产物；合成数值不能用于论文。

<details>
<summary>历史接口与旧方法记录（以下不是当前运行说明，命令和组数已退役）</summary>

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

</details>
