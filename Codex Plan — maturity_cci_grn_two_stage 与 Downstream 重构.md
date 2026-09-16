# Codex Plan: Integrate `maturity_cci_grn_two_stage` and Refactor Downstream Architecture

## 0. 总目标

基于**当前项目代码**进行一次受控整合。

队友更新版代码位于：

```text
C:\Users\27139\OneDrive\Desktop\Huge Workplace\2025 Causality\shr_code.zip
```

该 ZIP **只能作为参考实现（reference implementation）**，不能整体覆盖当前项目。

本次改造有两个主要目标：

1. 将队友新增的 **two-stage training + 新 Loss 系统**作为一个新的独立最优粗粒化方法注册：
   ```text
   maturity_cci_grn_two_stage
   ```
   它必须严格继承现有：
   ```text
   complete_combined_coarse_maturity_cci_grn
   ```
   的输入特征、CCI、GRN、maturity 构建逻辑，只改变训练策略和优化目标。

2. 接纳 `shr_code.zip` 中新的 downstream analysis 数学实现，尤其包括：
   - dynamic closure
   - information retention / leakage
   - cross-fit closure
   - null model
   - GRN perturbation
   - soft-native downstream
   - 以及队友已经同步重构的其他 downstream analysis

   但是：

   **当前项目原版 visualization 的视觉风格必须保留。**

最终将当前：

```text
mignet_ce/visualization/
```

整体移除，并将 visualization 系统迁移到：

```text
mignet_ce/downstream/analysis/visualization/
```

最终必须形成明确的：

```text
coarse frontend
    ↓
training
    ↓
analysis
    ↓
visualization
```

分层结构。

---

# 1. 本次修改的绝对原则

## 1.1 当前项目是唯一基线

不要：

```text
unzip shr_code.zip -> overwrite repository
```

不要整体复制：

```text
shr_code/mignet_ce/
shr_code/scripts/
shr_code/wyt_deltaei_coarse_grain/
```

覆盖当前项目。

正确方式：

1. 将 `shr_code.zip` 解压到临时参考目录。
2. 对照当前项目逐文件审计。
3. 仅迁移本计划明确要求的实现。
4. 当前项目中与本任务无关的代码、配置、实验、文档不得修改。

---

## 1.2 三种 optimized coarse-graining 必须同时存在

本次修改后，不允许用 two-stage 覆盖旧方法。

正式保留：

```text
complete_combined_coarse
complete_combined_coarse_maturity_cci_grn
maturity_cci_grn_two_stage
```

语义分别为：

### Method 1

```text
complete_combined_coarse
```

基础 optimized coarse-graining。

保持当前项目原始训练语义。

---

### Method 2

```text
complete_combined_coarse_maturity_cci_grn
```

原有 biologically informed optimized coarse-graining。

输入包括：

```text
complete representation
+ maturity
+ CCI
+ GRN
```

保持当前项目原始单阶段训练语义。

---

### Method 3

```text
maturity_cci_grn_two_stage
```

新的正式方法。

其 feature/frontend 必须与：

```text
complete_combined_coarse_maturity_cci_grn
```

严格一致。

区别只能来自：

```text
two-stage optimization
+ dynamic closure objectives
+ information retention objectives
+ Keff constraints
+ new checkpoint selection
```

不能偷偷改变：

- feature construction
- maturity definition
- CCI construction
- GRN construction
- N / Native-V7 representation
- preprocessing
- spot ordering
- PIJ input convention
- NMF semantics
- biological prior semantics

因此科学上必须满足：

```text
Maturity+CCI+GRN
        |
        | same exact frontend
        |
        +--------------------------+
        |                          |
 single-stage                 two-stage
        |                          |
        v                          v
complete_combined_          maturity_cci_grn_
coarse_maturity_cci_grn     two_stage
```

这样 Method 2 vs Method 3 才能作为严格的训练策略对照。

---

# 2. 首先审计 `shr_code.zip`

在进行修改前，先建立 source-level diff。

重点比较：

```text
wyt_deltaei_coarse_grain/
mignet_ce/coarse_frontends/
mignet_ce/visualization/downstream/
scripts/
tests/
```

将差异分为：

```text
A. mathematical/method changes
B. downstream analysis changes
C. visualization-only changes
D. engineering/refactor changes
E. accidental/noise changes
```

忽略：

```text
__pycache__
*.pyc
.ipynb_checkpoints
temporary output
checkpoints
line-ending-only differences
```

不要把 CRLF/LF 变化视为代码修改。

---

# 3. 新增 `maturity_cci_grn_two_stage` coarse frontend

## 3.1 新文件

建议新增：

```text
mignet_ce/coarse_frontends/maturity_cci_grn_two_stage.py
```

不要复制一整份 maturity frontend 实现后独立维护。

优先让它直接 delegate 到：

```python
complete_combined_coarse_maturity_cci_grn.prepare
```

即：

```text
maturity_cci_grn_two_stage.prepare(request)
            |
            v
complete_combined_coarse_maturity_cci_grn.prepare(request)
```

确保两个方法生成完全相同的：

```text
PreparedCoarseInput
features_t
features_tp
CCI representation
GRN representation
maturity representation
unit ordering
metadata
```

如需在 metadata 中标记 method identity，可以增加非数学 metadata，但不得改变 feature values。

---

# 4. 注册新 frontend

修改：

```text
mignet_ce/coarse_frontends/registry.py
```

新增：

```text
maturity_cci_grn_two_stage
```

使：

```python
prepare_coarse_input(
    "maturity_cci_grn_two_stage",
    request,
)
```

合法。

同时更新：

```text
mignet_ce/coarse_frontends/__init__.py
```

如有必要。

---

# 5. 不允许 two-stage 污染现有方法

这是本任务最重要的工程要求之一。

`shr_code.zip` 当前已经将 two-stage training 直接写入通用：

```text
wyt_deltaei_coarse_grain/trainer.py
```

如果原样接纳，会导致旧方法也使用新的训练语义。

这是不允许的。

必须建立明确的 training-mode separation。

推荐：

```python
training_mode:
    "legacy_single_stage"
    "dynamic_closure_two_stage"
```

或者等价的强类型实现。

默认必须保持：

```text
legacy_single_stage
```

保证任何旧代码在没有显式请求 two-stage 时行为不变。

---

# 6. Trainer 推荐架构

建议将通用入口保持为：

```python
train_deltaei(...)
```

内部根据 training mode dispatch：

```text
train_deltaei()
    |
    +-- legacy_single_stage
    |       |
    |       +-- 原项目训练逻辑
    |
    +-- dynamic_closure_two_stage
            |
            +-- shr_code two-stage implementation
```

例如内部拆成：

```python
_train_single_stage(...)
_train_two_stage(...)
```

但不要为了重构而修改任何数学定义。

---

# 7. Legacy single-stage 必须严格保存

以下两个方法默认使用：

```text
complete_combined_coarse
complete_combined_coarse_maturity_cci_grn
```

训练行为必须与当前项目修改前一致。

包括：

- Loss composition
- ΔEI objective
- usage regularization
- checkpoint selection
- epoch behavior
- default hyperparameters
- output semantics

不能因为接入队友 trainer 而改变历史结果。

如果当前原版 trainer 与 shr trainer 差异很大：

**保留原 trainer loop 作为 `_train_single_stage()`，不要反向模拟。**

---

# 8. `maturity_cci_grn_two_stage` 使用队友 two-stage 训练

只有：

```text
maturity_cci_grn_two_stage
```

默认启用：

```text
dynamic_closure_two_stage
```

它的核心语义按照 `shr_code.zip`。

---

# 9. Stage 1 语义

Stage 1 负责获得高 EI coarse-graining solution。

保持队友实现中的主要设计：

```text
Stage 1:
    maximize Delta EI
    + alignment
    + maturity/biological constraints
    + structural regularization
```

Stage 1 最优 checkpoint：

```text
best_ei.pt
```

Stage 1 结束后：

```text
reload best_ei.pt
```

再进入 Stage 2。

Stage 2 不能直接从 Stage 1 最后一个 epoch 开始。

---

# 10. Stage 2 语义

Stage 2 的科学目标是：

```text
preserve high causal emergence
while improving dynamical sufficiency
```

即：

```text
high ΔEI
    +
dynamic closure
    +
retained future information
    +
within-state future consistency
    +
between-state future separation
    +
effective macro-state constraints
```

按照 `shr_code.zip` 中现有数学定义迁移：

```text
L_closure
L_within
L_inter
L_retain
L_retain_floor
L_keff
EI floor
dead-prototype / usage constraint
```

不要自行重新设计这些 Loss。

---

# 11. Two-stage Loss 不得进入原 maturity 方法

必须有 regression test 验证：

```text
complete_combined_coarse_maturity_cci_grn
```

调用时：

```text
L_closure = not active
L_within = not active
L_inter = not active
L_retain = not active
two-stage checkpoint selection = not active
```

而：

```text
maturity_cci_grn_two_stage
```

才启用它们。

---

# 12. 修复 `shr_code` 中 two-stage checkpoint 已知问题

在迁移 teammate implementation 时，不要机械复制已经发现的实现错误。

## 12.1 `checkpoint_keff_min`

`shr_code` 中定义了：

```text
checkpoint_keff_min
```

但实际 joint checkpoint eligibility 没有真正检查它。

应让 joint eligibility 明确包含：

```text
Delta EI floor passed
AND retained information floor passed
AND Keff floor passed
```

也就是：

```text
eligible_joint =
    ei_ok
    and retain_ok
    and keff_ok
```

---

## 12.2 `best_joint.pt` 不得被 fallback 覆盖

当前 `shr_code` 存在风险：

```text
best valid joint checkpoint
        ↓
later epoch has higher ΔEI
        ↓
fallback path overwrites best_joint.pt
```

这是错误的。

必须独立维护：

```text
best_joint.pt
best_stage2_fallback.pt
```

或者在内存中分别追踪后最终一次性写入。

规则：

如果存在 eligible joint checkpoint：

```text
best_joint.pt
```

必须永远对应真正最佳 joint score。

只有：

```text
no Stage 2 epoch satisfies joint eligibility
```

时才允许使用 fallback。

fallback 不得覆盖已经存在的 valid joint checkpoint。

---

# 13. 统一 two-stage defaults

当前 `shr_code` 中至少存在：

```text
trainer default
CLI default
```

对 EI retention ratio 不一致的问题。

不要保留这种双重默认。

执行时：

1. 检查 `shr_code.zip` 中 teammate 已产生结果的：
   ```text
   config.json
   summary.json
   ```
   如果 ZIP 内包含正式结果配置，则以实际正式结果使用的数值为优先依据。

2. 如果没有结果配置可判断，则以 core `WYTDeltaEIConfig` 为 canonical source。

3. CLI 不再拥有独立默认数学参数。

CLI 应从统一 profile/config 获取默认值。

目标：

```text
one source of truth
```

---

# 14. 方法级 training profile

建议建立一个清晰 mapping，例如：

```text
complete_combined_coarse
    -> legacy_single_stage

complete_combined_coarse_maturity_cci_grn
    -> legacy_single_stage

maturity_cci_grn_two_stage
    -> dynamic_closure_two_stage
```

不要在多个 script 中重复：

```python
if method == ...
```

优先集中到一个 method specification / registry。

可以扩展 coarse frontend registry，例如引入：

```python
CoarseMethodSpec
```

包含：

```text
name
prepare
training_mode
requires_maturity
requires_cci
requires_grn
```

但：

**只在能够明显减少重复逻辑时做。**

不要为了“架构漂亮”过度重构整个 coarse frontend subsystem。

---

# 15. Downstream 总体重构目标

将当前：

```text
mignet_ce/visualization/downstream/
```

以及相关 downstream 代码重构为：

```text
mignet_ce/downstream/
└── analysis/
```

并将 visualization 放到：

```text
mignet_ce/downstream/analysis/visualization/
```

最终推荐：

```text
mignet_ce/
├── coarse_frontends/
│   ├── complete_combined_coarse.py
│   ├── complete_combined_coarse_maturity_cci_grn.py
│   ├── maturity_cci_grn_two_stage.py
│   └── registry.py
│
├── downstream/
│   ├── __init__.py
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── mappings.py
│       ├── metrics.py
│       ├── preparation.py
│       ├── reporting.py
│       ├── workflow.py
│       │
│       ├── dynamic_closure/
│       │   ├── __init__.py
│       │   ├── analysis.py
│       │   └── optimal.py
│       │
│       ├── determinism_degeneracy/
│       │   └── analysis.py
│       │
│       ├── fate_path/
│       │   └── analysis.py
│       │
│       ├── grn_cci/
│       │   └── analysis.py
│       │
│       ├── null_model/
│       │   └── analysis.py
│       │
│       ├── spatial/
│       │   └── analysis.py
│       │
│       ├── grn_perturbation/
│       │   ├── analysis.py
│       │   ├── baseline.py
│       │   ├── model_artifact.py
│       │   ├── operators.py
│       │   ├── propagation.py
│       │   └── validation.py
│       │
│       └── visualization/
│           ├── __init__.py
│           ├── style.py
│           ├── common.py
│           ├── plots.py
│           │
│           ├── dynamic_closure/
│           │   └── plots.py
│           ├── determinism_degeneracy/
│           │   └── plots.py
│           ├── fate_path/
│           │   └── plots.py
│           ├── grn_cci/
│           │   └── plots.py
│           ├── null_model/
│           │   └── plots.py
│           ├── spatial/
│           │   └── plots.py
│           └── grn_perturbation/
│               └── plots.py
```

具体文件可以按照现有模块需要小幅调整，但必须保持：

```text
analysis logic
        !=
visualization logic
```

---

# 16. 删除旧 `mignet_ce/visualization`

迁移完成后：

```text
mignet_ce/visualization/
```

不应继续作为正式 package 存在。

所有：

```python
from mignet_ce.visualization...
```

必须更新。

迁移后统一使用：

```python
from mignet_ce.downstream.analysis...
```

或：

```python
from mignet_ce.downstream.analysis.visualization...
```

不要留下两套 visualization package 长期并存。

如果为了迁移测试需要临时 compatibility import，可以在开发过程中使用，但最终提交前删除。

---

# 17. Analysis 来源优先级

Downstream analysis 数学实现：

**以 `shr_code.zip` 为主要参考。**

尤其接受：

```text
soft-native MappingRecord
dynamic closure information decomposition
I_available
I_retained
I_leakage
closure quality
within-dynamics JS
closure JS
Q operational gap
soft cross-fit
soft null model
soft spatial statistics
soft GRN/CCI pooling
soft fate alignment
GRN perturbation framework
```

不要为了兼容旧 visualization 而恢复 hard assignment downstream。

---

# 18. Soft-native 是正式 downstream contract

Optimized coarse-graining 必须使用：

```text
S_t
S_tp
```

原生 soft assignment。

禁止：

```python
argmax(S)
```

后再作为默认 downstream representation。

Seurat K40/K150 可以继续通过：

```text
one-hot S
```

进入同一套接口。

这样自然 baseline 和 optimized methods 都使用：

```text
N x K assignment matrix
```

作为统一 representation。

---

# 19. Unified mappings 扩展为 5 个

当前 teammate downstream 大致为：

```text
Seurat K150
Seurat K40
complete
maturity
```

现在必须新增：

```text
maturity_cci_grn_two_stage
```

最终：

```text
Seurat K150
Seurat K40
Optimized complete_combined_coarse
Optimized complete_combined_coarse_maturity_cci_grn
Optimized maturity_cci_grn_two_stage
```

修改：

```text
config.py
mappings.py
preparation.py
reporting.py
workflow.py
相关 plotting code
相关 tests
```

不要硬编码“4 mappings”。

搜索并处理类似：

```text
four-representation
4 mappings
2 methods x 3 pairs
expected six DeltaEI jobs
fixed 2x3 four-representation
```

等旧假设。

---

# 20. Optimized coarse-graining cache 应变为 3 methods × 3 adjacent pairs

正式 optimized jobs：

```text
complete_combined_coarse
complete_combined_coarse_maturity_cci_grn
maturity_cci_grn_two_stage
```

每个：

```text
11.5 -> 12.5
12.5 -> 13.5
13.5 -> 14.5
```

所以：

```text
3 x 3 = 9 optimized jobs
```

不能再保留：

```python
if len(outputs) != 6
```

这类旧 hard-coded contract。

---

# 21. 不同 method 的 checkpoint contract 必须区分

旧 single-stage 方法可能使用：

```text
best_model.pt
```

two-stage 使用：

```text
best_ei.pt
best_joint.pt
```

Downstream preparation 不得强迫所有方法都存在：

```text
best_joint.pt
```

应根据 method/training profile 检查。

例如：

### legacy single-stage

要求：

```text
config.json
input_manifest.json
feature_manifest.json
summary.json
best_model.pt
S_t.npy
S_tp.npy
PIJ_micro_train.npy
PIJ_macro_train.npy
assignments_t.csv
assignments_tp.csv
```

### two-stage

要求：

```text
config.json
input_manifest.json
feature_manifest.json
summary.json
best_ei.pt
best_joint.pt
S_t.npy
S_tp.npy
PIJ_micro_train.npy
PIJ_macro_train.npy
assignments_t.csv
assignments_tp.csv
```

最终 downstream 尽量依赖：

```text
S_t.npy
S_tp.npy
PIJ_micro_train.npy
PIJ_macro_train.npy
summary.json
```

而不是直接依赖 checkpoint 内部结构。

---

# 22. Cache manifest 必须记录训练方法

每个 optimized cache 至少记录：

```text
method
frontend
training_mode
objective_version
k
epochs
seed
NMF profile
maturity configuration
two-stage hyperparameters if applicable
```

例如：

```text
training_mode:
    legacy_single_stage
```

或：

```text
training_mode:
    dynamic_closure_two_stage
```

防止历史 single-stage cache 被 two-stage 错误复用。

---

# 23. 不覆盖旧输出

如果已有 cache 与新 manifest 不一致：

```text
DO NOT overwrite
```

应：

```text
raise clear error
```

要求用户使用新的 output/cache root。

不能静默重训练并覆盖旧结果。

---

# 24. Dynamic Closure：接纳 teammate mathematics

迁移 `shr_code` 中新的 dynamic closure implementation。

正式定义应继续使用 soft assignment。

核心：

```text
R = P_micro @ S_tp
```

从 source assignment 诱导：

```text
Q_induced
```

并计算：

```text
I_available
I_retained
I_leakage
closure_quality
within_dynamics_js
closure_js
q_operational_gap
```

保留信息论 identity validation：

```text
I_available
≈ I_retained + I_leakage
```

并设置合理 numerical tolerance。

---

# 25. Cross-fit Closure

接纳 teammate soft-native crossfit。

训练 fold 学：

```text
Q_train
```

test fold：

```text
R_hat_test = S_test @ Q_train
```

不要恢复旧 hard-state singleton-specific algorithm。

同步修复 teammate tests 中仍期待旧 hard assignment 行为的测试。

---

# 26. Null Model

采用 teammate 新版 null semantics。

保持 soft state mass 的同时破坏：

```text
spot <-> macro membership correspondence
```

source / target assignment permutation 后重新计算相关 dynamics。

明确保存：

```text
observed
null distribution
effect size
p-value
z-score
seed
repeat count
```

visualization 不重新生成 null samples。

---

# 27. GRN Perturbation

采用 `shr_code` 中新的：

```text
grn_perturbation/
```

完整分析框架。

包括：

```text
operators
propagation
analysis
validation
baseline/model artifact handling
```

不要恢复旧版：

```text
visualization/downstream/perturbation
```

中的旧 perturbation mathematics。

旧 perturbation 只作为：

```text
visual style/layout reference
```

使用。

---

# 28. GRN Perturbation 默认主模型改为新方法

`shr_code` 当前类似：

```python
METHOD = "complete_combined_coarse_maturity_cci_grn"
```

需要改造。

默认 perturbation 的 optimized model 应为：

```text
maturity_cci_grn_two_stage
```

但不要在深层 module 中硬编码。

推荐由：

```text
run_grn_perturbation.py
```

提供：

```text
--method
```

默认：

```text
maturity_cci_grn_two_stage
```

分析内部从 config 接收。

这样未来仍可运行：

```text
complete_combined_coarse_maturity_cci_grn
```

作为 perturbation control，而不用改源码。

---

# 29. Horizontal / Vertical propagation 保持 teammate 实现

不得改变 teammate 定义的：

```text
horizontal propagation
vertical propagation
horizontal -> vertical
vertical -> horizontal
```

及 12 类 perturbation operator 的数学语义。

本次任务重点是：

```text
architecture + integration + visualization
```

不是重新设计 perturbation experiment。

---

# 30. Visualization 的来源优先级与 analysis 相反

Analysis：

```text
shr_code implementation > old implementation
```

Visualization：

```text
current project old visualization > shr_code visualization
```

也就是说：

```text
new mathematics
+
old visual language
```

---

# 31. 当前 visualization 风格必须原样保留

重点保留当前项目原版：

```text
mignet_ce/visualization/downstream/style.py
```

以及旧 plots 中已经成熟的：

- fonts
- font sizes
- line widths
- marker size
- axes
- spines
- tick formatting
- legends
- panel spacing
- figure dimensions
- color philosophy
- grid behavior
- annotations
- significance labels
- export DPI
- PDF/SVG behavior
- layout
- subplot organization
- title style

不要为了“现代化”重新设计图。

---

# 32. 原有 mapping 颜色不改变

现有：

```text
K150
K40
Complete
Maturity
```

的颜色、marker 保持不变。

新增：

```text
maturity_cci_grn_two_stage
```

需要一个第五个可区分的颜色/marker。

要求：

- 从当前项目既有视觉体系中选取
- 低饱和
- 与现有 palette 协调
- 不改变其他四种 mapping 的颜色
- 不使用随机 matplotlib 默认颜色

---

# 33. Visualization 只能消费 analysis outputs

建立明确 contract：

```text
analysis
    |
    | writes
    v
CSV / JSON / NPY / NPZ
    |
    | reads
    v
visualization
```

Plot functions 不应：

- 重新训练
- 重新生成 null
- 重新计算 closure
- 重新计算 perturbation
- 修改 assignment
- argmax soft assignment
- 重新构建 PIJ

Plotting layer 只负责：

```text
read
reshape for plotting
plot
annotate
export
```

---

# 34. 为 teammate 新指标制作旧 style visualization

如果 teammate downstream 新增了旧代码不存在的指标，例如：

```text
I_available
I_retained
I_leakage
closure_quality
within_dynamics_js
closure_js
q_operational_gap
Keff
perturbation propagation discrepancy
```

不要删除这些指标。

为它们新增 plot adapter。

风格必须模仿当前项目现有 figures，而不是另起新 style。

---

# 35. Dynamic Closure visualization

至少支持统一比较：

```text
Seurat K150
Seurat K40
Complete
Maturity+CCI+GRN
Maturity+CCI+GRN Two-stage
```

在：

```text
11.5 -> 12.5
12.5 -> 13.5
13.5 -> 14.5
```

上的结果。

核心指标优先：

```text
closure_quality
I_retained
I_leakage
within_dynamics_js
closure_js
crossfit closure
```

不要只为 two-stage 画单独图。

要支持统一 benchmark。

---

# 36. Perturbation visualization

队友已经完成 analysis 但 visualization 不完整，因此重点补齐：

```text
strength-response
operator comparison
time-pair comparison
horizontal/vertical propagation comparison
effect size ranking
summary heatmap
repeat uncertainty
```

具体生成哪些 panel 以 teammate output schema 为准。

视觉设计优先复用当前项目旧 perturbation plot 的：

```text
layout
typography
axes
color discipline
legend
annotation
export
```

但绝不能调用旧 perturbation math。

---

# 37. 将旧 plotting modules 迁移到新位置

原：

```text
mignet_ce/visualization/downstream/*/plots.py
```

迁移为：

```text
mignet_ce/downstream/analysis/visualization/*/plots.py
```

analysis module 不能再与 plots.py 混放。

例如：

```text
mignet_ce/downstream/analysis/dynamic_closure/analysis.py
```

对应：

```text
mignet_ce/downstream/analysis/visualization/dynamic_closure/plots.py
```

---

# 38. 处理原 `mignet_ce/visualization` 中非 downstream 文件

当前类似：

```text
mignet_ce/visualization/common.py
mignet_ce/visualization/ei_existence.py
```

也要迁移到：

```text
mignet_ce/downstream/analysis/visualization/
```

不要留下旧 `mignet_ce/visualization` package。

如果 `ei_existence.py` 科学上属于 main analysis 而不是纯 plotting，则进行最小职责拆分：

```text
analysis calculation
-> mignet_ce/downstream/analysis/

plotting
-> mignet_ce/downstream/analysis/visualization/
```

不要仅为了目录迁移破坏功能。

---

# 39. Scripts 更新

全面检查：

```text
scripts/run_unified_downstream_analysis.py
scripts/run_downstream_analysis.py
scripts/render_unified_downstream_figure.py
scripts/run_grn_perturbation.py
scripts/plot_perturbation_metrics.py
scripts/run_wyt_deltaei_coarse_grain.py
```

更新 import 和方法选择。

要求：

### `run_wyt_deltaei_coarse_grain.py`

支持：

```text
--method maturity_cci_grn_two_stage
```

自动选择 two-stage。

旧 method 仍自动选择 legacy single-stage。

---

### `run_unified_downstream_analysis.py`

默认纳入五种 mappings。

---

### `run_grn_perturbation.py`

默认：

```text
--method maturity_cci_grn_two_stage
```

---

### `render_unified_downstream_figure.py`

只能调用新的：

```text
mignet_ce.downstream.analysis.visualization
```

---

# 40. 不再使用旧 import

执行：

```text
grep/search:
mignet_ce.visualization
```

正式源码中应为 0。

测试、README、scripts 中也同步更新。

不要留下死 import。

---

# 41. README / 文档同步

更新 downstream README，解释新架构：

```text
frontend
training
analysis
visualization
```

说明：

### Optimized methods

```text
complete_combined_coarse
complete_combined_coarse_maturity_cci_grn
maturity_cci_grn_two_stage
```

说明：

```text
maturity_cci_grn_two_stage
```

和 maturity 版本使用完全相同 frontend，仅训练策略不同。

---

# 42. 测试：frontend identity

新增测试。

对于完全相同 request：

```python
prepare_coarse_input(
    "complete_combined_coarse_maturity_cci_grn",
    request,
)
```

与：

```python
prepare_coarse_input(
    "maturity_cci_grn_two_stage",
    request,
)
```

必须验证：

```text
same units
same feature shapes
same feature values
same CCI-derived values
same GRN-derived values
same maturity values
same PIJ inputs
```

允许：

```text
method metadata
```

不同。

这是整个新方法最重要的 scientific regression test。

---

# 43. 测试：training isolation

验证：

```text
complete
maturity
```

仍进入：

```text
legacy_single_stage
```

验证：

```text
maturity_cci_grn_two_stage
```

进入：

```text
dynamic_closure_two_stage
```

---

# 44. 测试：two-stage smoke

使用极小 synthetic fixture：

```text
few spots
few states
2-5 epochs
CPU
```

验证：

```text
Stage 1 runs
best_ei.pt produced
Stage 2 reloads best_ei
closure losses finite
I_available finite
I_retained finite
Keff finite
best_joint.pt produced
S_t rows sum to 1
S_tp rows sum to 1
PIJ rows normalized
```

不要在 test suite 中运行 1500 epochs。

---

# 45. 测试：checkpoint eligibility

构造测试确保：

```text
EI pass + retain pass + Keff fail
```

时：

```text
NOT eligible
```

确保：

```text
valid best_joint
```

不会被之后的 fallback epoch 覆盖。

---

# 46. 测试：legacy reproducibility

如果已有原 trainer fixture/golden numbers：

修改前后：

```text
complete_combined_coarse
complete_combined_coarse_maturity_cci_grn
```

在固定：

```text
seed
data
epochs
config
```

下保持一致。

至少验证：

```text
loss history
best epoch
Delta EI
S_t
S_tp
checkpoint type
```

在合理数值 tolerance 内一致。

---

# 47. 测试：5-mapping downstream

统一 downstream fixture 必须包含：

```text
K150
K40
Complete
Maturity
Two-stage
```

检查：

```text
load_all_mapping_records()
```

返回：

```text
5 mappings × 3 adjacent pairs = 15 records
```

---

# 48. 测试：soft-native contract

对于 optimized mapping：

禁止默认调用：

```text
argmax
```

测试至少放一个明显 soft assignment：

```text
[0.45, 0.40, 0.15]
```

确保 downstream 保留概率信息。

---

# 49. 测试：dynamic closure information identity

验证：

```text
I_available
≈ I_retained + I_leakage
```

在设定 numerical tolerance 内成立。

---

# 50. 测试：visualization 不修改结果

Plot test 应：

1. 准备固定 analysis output table。
2. 运行 plotting。
3. 验证 figure 成功保存。
4. 验证输入文件 hash / dataframe 不改变。

visualization 不得写回分析结果。

---

# 51. 测试：旧 style preservation

如果当前项目有 figure regression 或 style test，保留。

没有的话至少检查：

```text
font settings
figure DPI
export format
original mapping colors
markers
style context
```

未被改变。

不要使用 pixel-perfect screenshot regression，除非项目本来已经有。

---

# 52. 修复 teammate 当前失败的相关测试

之前 teammate 版本存在至少两类 stale test：

```text
hard-state singleton crossfit expectation
old MappingRecord hs/ht fields
```

正式版本全部更新到：

```text
source_assignment
target_assignment
soft-native semantics
```

不要为了让旧测试通过而恢复旧 downstream math。

测试必须服从新的正式方法定义。

---

# 53. Import-cycle 检查

新目录：

```text
downstream/analysis/
downstream/analysis/visualization/
```

容易形成循环依赖。

明确约束：

```text
analysis NEVER imports visualization
```

允许：

```text
visualization imports analysis schemas/constants
```

但尽量 visualization 只消费结果文件/dataframe。

---

# 54. Full-scale 正式运行不是本 Codex 修改任务的一部分

Codex 完成代码后：

不要自动运行：

```text
1500 epochs × 9 optimized jobs
200 perturbations × ...
full embryo pipeline
```

只运行：

```text
unit tests
integration smoke tests
small synthetic tests
CLI --help/import tests
```

正式 A100 full run 由用户后续手动执行。

---

# 55. 不要修改的内容

除非本计划明确要求，不要改：

```text
data_factory/
GRN construction semantics
CCI generation
spot preprocessing
Seurat K40/K150 generation
NG-KLot mathematics
EI definition
maturity generation mathematics
feature projection mathematics
complete frontend mathematics
original single-stage objective
```

不要做“顺手优化”。

---

# 56. 不做无关 cleanup

不要：

- 全仓库格式化
- 改变量名只为了美观
- 修改所有 docstring
- 重写 logging system
- 改 CLI 风格
- 删除用户现有脚本
- 修改 data path convention
- 修改默认 output root
- 修改数学公式
- 修改 random seed

只做完成本计划所需的最小结构重构。

---

# 57. Git / working tree 安全

开始前检查：

```text
git status
```

不得：

```text
git reset --hard
git clean -fd
```

不得覆盖当前用户未提交修改。

如果目标文件已经有用户修改：

在现有内容基础上 merge。

---

# 58. 最终目标目录检查

最终必须不存在：

```text
mignet_ce/visualization/
```

必须存在：

```text
mignet_ce/downstream/analysis/
mignet_ce/downstream/analysis/visualization/
```

并且：

```text
analysis modules
```

与：

```text
plotting modules
```

职责清晰。

---

# 59. 最终功能检查

必须可以：

```bash
python scripts/run_wyt_deltaei_coarse_grain.py --help
python scripts/run_unified_downstream_analysis.py --help
python scripts/run_grn_perturbation.py --help
python scripts/render_unified_downstream_figure.py --help
```

并成功 import。

新 method 必须出现在：

```text
--method
```

choices 中。

---

# 60. 最终 automated audit

执行类似：

```text
pytest
```

如果全量 suite 很大，至少首先运行所有涉及：

```text
coarse_frontends
wyt_deltaei
two_stage
dynamic_closure
downstream
mappings
null
spatial
grn_perturbation
visualization
```

的测试。

然后尽可能运行完整 suite。

所有 failure 必须分类：

```text
real regression
stale old test
environment/data unavailable
```

不能直接删除失败测试。

---

# 61. 最终人工代码审计

完成后重新搜索：

```text
mignet_ce.visualization
complete_combined_coarse_maturity_cci_grn hard-coded perturbation target
UNIFIED_MAPPINGS assumptions
four-representation
len(...) == 4
len(outputs) == 6
2 methods x 3
best_joint.pt required for legacy method
argmax(source_assignment)
argmax(target_assignment)
```

逐一确认是否仍存在不合理旧假设。

---

# 62. 最终必须向用户报告的内容

Codex 完成后不要只说“done”。

输出一个清晰报告，包括：

## A. 新增文件

列出：

```text
maturity_cci_grn_two_stage
new downstream package
new visualization package
new tests
```

## B. 修改文件

逐文件说明修改目的。

## C. 删除/迁移文件

重点报告：

```text
mignet_ce/visualization/
```

迁移到了哪里。

## D. 方法隔离证明

明确说明：

```text
complete -> legacy
maturity -> legacy
two_stage -> two-stage
```

## E. Frontend identity

说明测试是否证明：

```text
maturity
```

和：

```text
maturity_cci_grn_two_stage
```

输入完全一致。

## F. Downstream

说明哪些模块来自 teammate 新数学语义：

```text
dynamic closure
null
crossfit
spatial
fate
GRN/CCI
perturbation
```

## G. Visualization

说明：

```text
analysis logic = new
visual style = current project old style
```

## H. Bug fixes

至少说明：

```text
checkpoint_keff_min
best_joint fallback overwrite
default inconsistency
stale soft-native tests
```

是否修复。

## I. Tests

给出：

```text
passed
failed
skipped
```

以及失败原因。

---

# 63. 最终科学方法关系必须保持

最终代码表达的实验结构必须是：

```text
                    Complete features
                          |
                          v
               complete_combined_coarse
                  legacy single-stage


        Maturity + CCI + GRN features
                     |
             identical frontend
                     |
          +----------+----------+
          |                     |
          v                     v

complete_combined_         maturity_cci_grn_
coarse_maturity_cci_grn    two_stage

legacy single-stage        Stage 1:
                           high Delta EI
                               |
                               v
                           Stage 2:
                           dynamic closure
                           retained information
                           within consistency
                           inter separation
                           EI floor
                           Keff floor
```

这是本次重构最重要的科学约束。

---

# 64. 最终 downstream 关系

最终统一分析：

```text
K150
K40
Complete
Maturity+CCI+GRN
Maturity+CCI+GRN Two-stage
        |
        v
soft-native MappingRecord
        |
        +----------------------+
        |                      |
        v                      v
Dynamic Closure          Other Analysis
Information Retention    Null
Cross-fit                Spatial
                         Fate
                         GRN/CCI
                         Perturbation
        |                      |
        +----------+-----------+
                   |
                   v
       standardized result tables
                   |
                   v
mignet_ce/downstream/analysis/visualization
                   |
                   v
     original project visual style
```

---

# 65. Definition of Done

本任务只有同时满足以下条件才算完成：

- [ ] `maturity_cci_grn_two_stage` 已正式注册。
- [ ] 它与原 maturity+CCI+GRN frontend 数值完全一致。
- [ ] 原 `complete` 方法没有切换到 two-stage。
- [ ] 原 `maturity_cci_grn` 方法没有切换到 two-stage。
- [ ] 新方法独占 two-stage training。
- [ ] teammate two-stage Loss 已迁移。
- [ ] `checkpoint_keff_min` eligibility 已修复。
- [ ] `best_joint.pt` fallback overwrite 已修复。
- [ ] two-stage default 参数只有一个 source of truth。
- [ ] unified downstream 已扩展到 5 mappings。
- [ ] optimized formal jobs contract 已扩展到 9 jobs。
- [ ] downstream 使用 soft-native assignment。
- [ ] teammate dynamic closure 已接纳。
- [ ] teammate GRN perturbation 已接纳。
- [ ] perturbation 默认模型为 `maturity_cci_grn_two_stage`。
- [ ] visualization 不重新计算 analysis。
- [ ] 当前项目旧 visualization style 被保留。
- [ ] `mignet_ce/visualization/` 已完全迁移并删除。
- [ ] 新目录为 `mignet_ce/downstream/analysis/visualization/`。
- [ ] 所有 import 已更新。
- [ ] stale hard-assignment tests 已按新数学定义更新。
- [ ] unit/integration smoke tests 通过。
- [ ] 未运行昂贵 full-scale training。
- [ ] 未修改与任务无关的方法、数据或实验逻辑。
- [ ] 最终提交包含详细 change report 和 test report。

---

## 一句话执行原则

**以当前项目为基线；保留 Complete 和 Maturity 两个历史方法；将 teammate 的 two-stage Loss/训练作为完全继承 Maturity+CCI+GRN frontend 的第三个方法 `maturity_cci_grn_two_stage`；采用 teammate 的新 downstream analysis 数学语义，但保留当前项目原 visualization 的视觉语言，并将整个 visualization 系统迁移至 `mignet_ce/downstream/analysis/visualization/`。**