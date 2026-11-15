# 空间 domain 图

## 正式默认配置与下一轮八组搜索

当前 `formal_two_stage` 的默认权重已直接替换为：Stage 1/2 Closure=1，Dev=2，
EI constraint=1，Retain norm=1，Retain floor=1，Lgap=1。
方法名仍为 `formal_two_stage`，唯一损失协议仍为 `formal_two_stage_main`。
这是默认参数替换，没有增加新的训练协议，也不修改其他 two-stage 方法的默认权重。

公式（星号代表选定 Stage 1 checkpoint 的冻结参考值）：

```text
Stage 1 = −ΔEI + 2 Ldev + Lclosure + Lgap
Stage 2 = 2 Ldev + ω (Lclosure + Lretain + Lretain_floor) + LEI_constraint + Lgap

Lretain = −Iretained / max(Iavailable*, 1e−8)
Lretain_floor = max(0, 0.5 − Iretained / max(Iretained*, 1e−8))²
LEI_constraint = max(0, ΔEI* − ΔEI)²
```

Ldev 是源、目标两侧簇内 maturity/pseudotime 方差损失的均值。
ω 根据冻结的 Stage 1 信息状态确定：informative 时为 1，low-signal 时为 0.1。
训练 EI 保留比例仍为 1，checkpoint EI 保留比例仍为 0.85；
Retain floor 和 checkpoint retain 比例仍为 0.5。现有 checkpoint 筛选规则没有改变。

新训练默认使用三因素二水平全因子实验（`--experiment factorial`）：

| 配置 | Closure（两阶段相同） | Dev | Retain norm |
|---|---:|---:|---:|
| S1 | 0.5 | 1 | 0.5 |
| S2 | 0.5 | 1 | 1 |
| S3 | 0.5 | 2 | 0.5 |
| S4 | 0.5 | 2 | 1 |
| S5 | 1 | 1 | 0.5 |
| S6 | 1 | 1 | 1 |
| S7 | 1 | 2 | 0.5 |
| S8（正式默认、对照） | 1 | 2 | 1 |

全部固定 EI constraint=1、Retain floor=1、Lgap=1，沿用 Heart/Spot/K40、三个时间对。
Closure 的低水平为 0.5/0.5，与历史 Baseline 的 0.5/0.1 不同。
4 组 Stage 1 配置×3 时间对=12 次 Stage 1，8 组×3 时间对=24 次独立 Stage 2。
相同 Closure、Dev 的两个 Retain norm 水平共享模型、优化器、随机数状态及冻结参考值。
默认最多六个训练子进程，每个任务在自己的 Stage 1 就绪后立即进入 Stage 2，无全局阶段屏障。

先只读检查上一轮各惩罚项是否激活（原实验目录不改写）：

```bash
python -m mignet_ce.downstream.domain_visualization optimal \
  --mode audit \
  --history-root "output/domain_visualization/formal_single_factor_v1" \
  --output-root "output/domain_visualization/formal_factorial_v1"
```

审计读取各 run 的 `metrics.csv`，只在 Stage 2 统计原始 EI constraint、Retain floor 大于零的
epoch 比例和平均/最大惩罚，同时导出各加权贡献。无需训练或重新选择 checkpoint。
也可在训练命令中添加同一个 `--history-root`，训练前自动完成审计。

```bash
python -m mignet_ce.downstream.domain_visualization optimal \
  --mode train --experiment factorial \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --output-root "output/domain_visualization/formal_factorial_v1" \
  --epochs 1500 --seed 42 --device cuda --workers 6
```

训练完成后自动输出三个时间对的八列 domain 图，S8 放在首列并显示绝对值，其余列显示与
同时间对 S8 的差值。绘图沿用已有 domain 样式。新实验使用新目录，不将历史结果作为新参数直接续跑。
续跑保持完整命令一致，仅将 `--mode train` 改成 `--mode resume`。

```bash
python -m mignet_ce.downstream.domain_visualization optimal \
  --mode render \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --output-root "output/domain_visualization/formal_factorial_v1"
```

`render` 自动识别保存的实验清单；没有 experiment 字段的旧清单按历史六版本读取。
`preview` 默认显示八列，可用 `--experiment single_factor` 预览历史六列。每次导出追加新的 render_NNNN 目录。

综合报告包括：

- `metrics.csv`：ΔEI、CQ、I_available、I_retained、EI macro、Closure JS mean/median/P90、
  Ldev、HardK、KeffHard、HardK−KeffHard、Stage 1 参考值及 ω。原始值与相对 S8 的差值均保留，
  `comparison_reference=s8`；兼容的 `delta_vs_baseline_*` 字段在这轮指相对 S8。
- `penalty_activation.csv`：每组 Stage 2 的惩罚激活比例和各加权 loss 贡献。
- `factorial_effects.csv`：C、D、R 主效应及两两/三因素交互，定义为 `2 × mean(编码符号乘积 × 指标)`。
- `pareto_candidates.json`：逐时间对及跨三个时间对的候选。最大化各时间对 ΔEI、CQ，最小化
  Closure JS mean；跨时间对使用九维目标，不平均 CQ，不设未经确认的容忍阈值，不自动挑选唯一赢家。
  I_available、I_retained、occupancy 作为并列核查指标，保留所有配置的完整结果。

以下六版本说明为历史实验记录。其权重被显式固定用于复现和读取，不代表当前正式默认配置。

## 独立的自然图与最优粗粒化 pipeline

两个入口的配置、调度、指标和结果目录均属于本模块，不导入 loss_ablation。
共享的只有项目公共输入准备、formal_two_stage 训练引擎和原 domain 绘图样式。
下文旧入口仍兼容原有四方法图；新的八组实验与历史六版本实验均使用 `optimal` 入口。

### 自然 Spot / Seurat 图

```bash
python -m mignet_ce.downstream.domain_visualization natural \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --time-pair 12.5 13.5 \
  --output-root "output/domain_visualization/natural_v1"
```

沿用原来的三列输入尺度、两行阶段布局，不依赖训练结果。

### 历史六版本单因素实验

| 版本 | 相对 Baseline 的唯一改动 | Stage 1 分组 |
|---|---|---|
| baseline | 不改 | baseline |
| closure | 两阶段 Closure：0.5 / 0.1 → 1 / 1 | closure |
| development | Dev：1 → 2 | development |
| ei_constraint | EI constraint：5 → 1 | baseline |
| retain_norm | Retain norm：0.5 → 1 | baseline |
| retain_floor | Retain floor：2 → 1 | baseline |

所有版本基于 Baseline，互不累计。其余权重及保护参数保持一致，Lgap=1。
固定为心脏 Spot→K40，三个时间对 11.5→12.5、12.5→13.5、11.5→13.5。

```bash
python -m mignet_ce.downstream.domain_visualization optimal \
  --mode train --experiment single_factor \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --output-root "output/domain_visualization/formal_single_factor_v1" \
  --epochs 1500 --seed 42 --device cuda --workers 6
```

上述显式指定 single_factor 的历史复现命令从头运行六版本。每个“版本×时间对”是一个任务，最多六个训练子进程。
同一时间对内配置一致的 Stage 1 只训练一次，共享完整 checkpoint、优化器和随机数状态；
每个任务的 Stage 1 就绪后立即进入独立 Stage 2，不等其他组。因此共 9 次 Stage 1 和 18 次 Stage 2。
按 0.45 的阶段比例，1500 epochs 对应 Stage 1 675 epochs、Stage 2 825 epochs。
每个子进程将 BLAS/OpenMP 线程数设为 1，避免六进程重复占满 CPU 线程。

输出结构：`_prepared/` 保存三份输入缓存，`stage1/<group>/<pair>/` 保存共享起点，
`runs/<variant>/<pair>/` 保存独立结果，`run_config.json` / `run_manifest.json` 记录设置、校验和及完成状态。
每次训练尝试创建新的 `attempt_NNNN`，保留失败记录。续跑复用校验通过的完整尝试；
未完成的尝试会重新训练对应阶段，不从中间 epoch 恢复。

续跑时复制完整训练命令，仅把 `--mode train` 改为 `--mode resume`，其他输入与设置不变。
新增流程不导入或读取其他实验模块的运行目录。

### 重画与预览

```bash
python -m mignet_ce.downstream.domain_visualization optimal \
  --mode render \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --output-root "output/domain_visualization/formal_single_factor_v1"
```

训练完成后自动画图；`render` 只读取已完成且校验通过的 18 组结果，不训练、不重新选 checkpoint。
每次重画追加 `figures/render_NNNN/`，内含三个时间对的 PNG/PDF、`metrics.csv` 和绘图记录。
所有指标对应同一个选定的 `best_joint.pt`。

六列沿用当前 domain 图的颜色、字体、散点和原始组织几何。每列上排输入、下排最优粗粒化，左源右目标。
Baseline 下显示绝对值，其他五列显示“本版本 − 同时间对 Baseline”的带符号差值：
ΔEI、CQ、I_available、HardK、HardK − KeffHard。后两项按源/目标显示。
KeffHard 按训练输入单元的 hard assignment 频率计算，不能用投影后的 Spot 数量替代。
CSV 同时保留原值、差值、KeffHard、I_retained、选定 epoch 和 fallback 状态。

将 `--mode render` 改为 `--mode preview` 可以仅用真实坐标预览版式；
未训练的 domain 和指标明确标为 unavailable，不填示例指标。
同一输出目录已有的 `previews/` 可以保留，随后用 `--mode train` 启动首次训练。

## 原四方法入口（兼容）

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
