# Plan 5：修复 v2 Application 调度，使其恢复为单 T1→T2 工作流

## 1. 目标

本计划只修复 `code_v2` 中 application 层的调度与复用问题，使 v2 的实验语义重新与 v1 一致：

```text
一次运行
=
明确指定一个 T1/source sample
+
明确指定一个 T2/target sample
```

而不是当前 v2 的：

```text
自动扫描全部样本
→ 强制 B1 / C1 / D1 三对
→ 一次运行 3 个 T1→T2 pair
```

本计划必须保证：

- Plan 1 一次只处理一对 T1/T2
- Plan 2 只对这一对跑 Seurat K150/K40
- Plan 3 只对这一对跑一次 Optimal CG
- Plan 4 只对这一对做一次 biological cluster coarse-graining
- Spot PIJ / EI reference 全局只构造一次并复用
- CCI / GRN / Seurat / Optimal CG / EI / CQ 核心算法全部不改
- 所有修复仅发生在：

```text
mignet_ce/downstream/application/
```

---

# 2. 修复原则

本计划是 application orchestration bugfix，不是算法开发。

## 2.1 可以修改

只允许修改：

```text
mignet_ce/downstream/application/
```

主要包括：

```text
cli/prepare_data.py
data/factory_bridge.py
common/spot_reference.py
cli/run_seurat_cg.py
cli/run_optimal_cg.py
cli/run_biological_cluster.py
必要的 application config / paths / tests
```

---

## 2.2 不允许修改

禁止修改任何核心算法目录：

```text
data_factory/
mignet_ce/coarse_frontends/
mignet_ce/downstream/analysis/
wyt_deltaei_coarse_grain/
scripts/
```

特别禁止修改：

```text
COMMOT / CCI algorithm
GRN / GENIE3-style algorithm
Seurat K40/K150 implementation
shared-core directed NMF
NG cost
Sinkhorn
effective_information
information_closure_budget
WYT model
two-stage trainer
two-stage objective
macro builder
maturity algorithm
```

原则：

```text
算法 = 继续调用已有代码
application = 只修调度、路径、缓存、参数、日志、报告
```

---

# 3. 当前 v2 的核心问题

当前 v2 的主要问题不是算法错误，而是 application 调度与 v1 的实验语义不一致。

---

## 3.1 Plan 1 自动扫描 6 个样本

当前行为：

```text
input_data/
    ↓
discover_samples()
    ↓
自动发现全部符合规则的样本
    ↓
B1 / C1 / D1
```

然后硬编码：

```python
pairs = {
    "B1": [],
    "C1": [],
    "D1": [],
}
```

最终实际运行：

```text
B1:
T1_B1 → T2_B1

C1:
T1_C1 → T2_C1

D1:
T1_D1 → T2_D1
```

所以一次 `prepare_data` 实际处理：

```text
6 samples
3 pairs
```

而 v1 是：

```text
2 samples
1 pair
```

这是 v2 Plan 1 计算量明显变大的首要原因。

---

## 3.2 GRN 日志被 `capture_output=True` 吞掉

当前 application bridge 使用类似：

```python
subprocess.run(
    command,
    cwd=PROJECT_ROOT,
    text=True,
    capture_output=True,
    check=False,
)
```

导致 GRN 内部：

```text
Matched TF regulators
GENIE3 targets progress
warnings
stdout
stderr
```

全部在任务结束前不可见。

表现为：

```text
==> Plan 1 ...
```

之后终端长时间没有任何输出。

这不是 GRN 算法本身需要修改，而是 application 调用方式需要修复。

---

## 3.3 Plan 2 仍然写死 B1/C1/D1

即使 Plan 1 改成单 pair，如果 Plan 2 仍然：

```python
REPLICATES = ("B1", "C1", "D1")
```

或要求：

```text
6 samples
```

那么 Plan 2 会直接失败。

所以 Plan 2 必须同步改成：

```text
从 manifest 读取唯一 source
+
唯一 target
```

---

## 3.4 Plan 3 当前会跑三次 Optimal CG

当前语义类似：

```python
for replicate in ("B1", "C1", "D1"):
    run_training(...)
```

需要改成：

```text
一个 source
+
一个 target
→ 一次 Optimal CG
```

---

## 3.5 Plan 4 当前也按三对运行

需要恢复成：

```text
一个 source/target pair
→ 一次 biological cluster analysis
```

---

## 3.6 Spot reference 没有真正全局复用

当前可能存在：

```text
Plan 2 K150
→ build spot PIJ

Plan 2 K40
→ 再 build 一次 spot PIJ

Plan 4
→ 再 build 一次 spot PIJ
```

同一个 T1→T2 的 micro dynamics 被重复构造。

必须改成：

```text
output/common/spot_reference/
```

只生成一次。

Plan 2 K150/K40 和 Plan 4 全部读取同一份。

---

# 4. 修复后的总体流程

```text
                    INPUT DATA
                        │
             explicit source + target
                        │
                        ▼
                     PLAN 1
              prepare exactly 2 samples
                        │
                Spot CCI + Spot GRN
                        │
                        ▼
               common spot reference
                        │
                PIJ_spot + EI_spot
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
       PLAN 2         PLAN 3        PLAN 4
   Seurat K150/K40   Optimal CG   Biological Map
          │             │             │
     ┌────┴────┐        │             │
     ▼         ▼        ▼             ▼
   K150       K40    one training   one fixed CG
     │         │        │             │
     ▼         ▼        ▼             ▼
  ΔEI+CQ    ΔEI+CQ   ΔEI+CQ        ΔEI+CQ
```

---

# 5. 修改一：`cli/prepare_data.py`

## 5.1 删除自动三 replicate 调度

删除/废弃：

```text
自动把全部 samples 分到 B1 / C1 / D1
pairs = {"B1": [], "C1": [], "D1": []}
要求每个 replicate 必须有 2 samples
for replicate in pairs:
    run_cci(...)
    run_grn(...)
```

---

## 5.2 新增显式 source / target 参数

CLI 增加：

```text
--source-sample
--target-sample
```

推荐设为 required。

例如：

```bash
python -m mignet_ce.downstream.application.cli.prepare_data \
  --input-root "$PWD/mignet_ce/downstream/application/input_data" \
  --output-root "$PWD/mignet_ce/downstream/application/output" \
  --source-sample <T1_SAMPLE_ID> \
  --target-sample <T2_SAMPLE_ID>
```

---

## 5.3 sample discovery 仍然可以复用

不需要重写 `paths.py` 的全部发现逻辑。

可以继续：

```python
discover_samples(...)
```

但只允许寻找：

```text
source_sample
target_sample
```

而不是无参数扫描全部 sample 后自动组 pair。

逻辑：

```text
discover source
discover target
↓
必须刚好得到 2 samples
↓
assign role
source
target
```

---

## 5.4 增加 role 字段

Plan 1 manifest 中每个 sample 明确记录：

```text
role = source
```

或：

```text
role = target
```

后续 Plan 2/3/4 不再根据：

```text
B1/C1/D1
071/072
day7/day21
```

猜测 source/target。

---

## 5.5 Plan 1 只调用一次 CCI / GRN

最终：

```python
sample_names = [
    source.sample_id,
    target.sample_id,
]
```

只调用：

```text
run_cci(... sample_names=[source, target])
run_grn(... sample_names=[source, target])
```

禁止再循环 replicate。

---

# 6. 修改二：`data/factory_bridge.py`

## 6.1 目的

修复长任务完全无日志的问题。

---

## 6.2 生产执行时禁止 `capture_output=True`

当前：

```python
completed = runner(
    command,
    cwd=PROJECT_ROOT,
    text=True,
    capture_output=True,
    check=False,
)
```

生产运行改为让子进程继承当前 terminal：

```text
stdout → terminal
stderr → terminal
```

这样 CCI/GRN 内部的：

```text
tqdm
Matched TFs
warnings
progress
```

全部实时显示。

---

## 6.3 保留测试可注入 runner

为了不破坏单元测试，`_run()` 仍可接受 injected runner。

建议：

```text
真实 subprocess.run：
不 capture

测试 fake runner：
按测试需要返回 mock CompletedProcess
```

不要为了测试重新把正式运行日志吞掉。

---

## 6.4 执行前打印命令

至少输出：

```text
[application] Running GRN:
python ... 07_run_grn_spot.py ...

[application] Running CCI:
python ... 04_run_cci_spot_commot.py ...
```

这样用户可以直接复制单独 debug。

---

# 7. 修改三：统一 `common/spot_reference.py`

## 7.1 目标

一对 T1/T2 只构造一次：

```text
PIJ_spot
EI_spot
source_spots
target_spots
```

---

## 7.2 唯一输出目录

固定：

```text
<output_root>/common/spot_reference/
```

禁止：

```text
seurat_cg/k150/.../spot_reference
seurat_cg/k40/.../spot_reference
biological_cluster/.../spot_reference
```

各自重复建立。

---

## 7.3 权威算法继续复用

内部仍调用现有：

```python
from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse import (
    prepare as prepare_complete_combined,
)
```

禁止自己实现 PIJ。

---

## 7.4 缓存判断

如果以下已经存在：

```text
PIJ_spot.npy
source_spots.csv
target_spots.csv
spot_reference_summary.json
spot_reference_provenance.json
```

并且 provenance 与当前：

```text
source H5AD
target H5AD
CCI
GRN
参数
```

一致，则直接复用。

否则重新生成。

---

# 8. 修改四：`cli/run_seurat_cg.py`

## 8.1 删除 replicate 逻辑

删除：

```text
REPLICATES = ("B1", "C1", "D1")
len(samples) == 6
for replicate in ...
```

---

## 8.2 从 manifest 找唯一 source/target

统一：

```text
source = role == "source"
target = role == "target"
```

要求：

```text
exactly 1 source
exactly 1 target
```

---

## 8.3 K150 / K40 仍然都跑

Plan 2 保持：

```text
Seurat K150
Seurat K40
```

对同一 source/target pair：

```text
source → target
```

分别执行。

---

## 8.4 Spot reference 只读取 common cache

改成：

```text
common/spot_reference/
```

只 build/load 一次。

然后：

```text
PIJ_spot
      │
 ┌────┴────┐
 ▼         ▼
K150      K40
```

K150/K40 不再分别调用 `build_spot_reference()`。

---

## 8.5 Seurat / CCI / GRN 算法不变

仍然直接调用：

```text
03_run_seurat_domains.R
run_cci_layer_commot.py
run_grn_layer.py
prepare_complete_stage
prepare_ngklot_pair
information_closure_budget
effective_information
```

本计划不改算法。

---

## 8.6 Plan 2 输出去掉 replicate 层

从类似：

```text
seurat_cg/k150/pairs/B1/
```

改成：

```text
seurat_cg/k150/
```

以及：

```text
seurat_cg/k40/
```

无需多一层 replicate。

---

# 9. 修改五：`cli/run_optimal_cg.py`

## 9.1 删除三 replicate training

删除：

```python
for replicate in ("B1", "C1", "D1"):
```

以及相关 pair discovery。

---

## 9.2 只跑唯一 source/target

从 Plan 1 manifest 读取：

```text
source
target
```

只调用一次：

```text
maturity_cci_grn_two_stage
```

---

## 9.3 K 改为 CLI 参数

如果当前写死：

```python
k=40
```

改为：

```text
--k
```

推荐：

```text
default=40
```

使用：

```bash
python -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --prepared-root ... \
  --output-root ... \
  --k 40
```

provenance 必须记录：

```text
K
Keff_t
Keff_tp
```

注意：

```text
K != Keff
```

---

## 9.4 Optimal CG 核心完全不改

仍然调用：

```text
scripts/build_maturity_proxy_from_h5ad.py
scripts/run_wyt_deltaei_coarse_grain.py
--method maturity_cci_grn_two_stage
```

禁止修改 trainer / loss / macro builder。

---

## 9.5 输出去掉 B1/C1/D1

最终：

```text
optimal_cg/
├── maturity/
├── training/
├── causal_emergence_summary.json
├── dynamical_closure_summary.json
└── provenance.json
```

而不是：

```text
training/B1
training/C1
training/D1
```

---

# 10. 修改六：`cli/run_biological_cluster.py`

## 10.1 删除 replicate loop

从：

```text
B1
C1
D1
```

改为：

```text
source
target
```

---

## 10.2 复用 common spot reference

禁止再次计算 Spot PIJ。

直接：

```text
load <output_root>/common/spot_reference/
```

---

## 10.3 Biological cluster maps

只读取：

```text
source cluster map
target cluster map
```

构造：

```text
S_source
S_target
```

然后调用现有：

```text
information_closure_budget
effective_information
state_level_ei
```

算法不变。

---

# 11. 修改七：manifest / provenance 统一

Plan 1 的主 manifest 建议改成：

```json
{
  "source_sample": {
    "sample_id": "...",
    "role": "source",
    "timepoint": "..."
  },
  "target_sample": {
    "sample_id": "...",
    "role": "target",
    "timepoint": "..."
  },
  "factory_options": {
    "...": "..."
  }
}
```

Plan 2/3/4 一律只依据：

```text
role = source
role = target
```

不要再次通过文件名推断 pair。

---

# 12. 修复后的 CLI

## Plan 1

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m mignet_ce.downstream.application.cli.prepare_data \
  --input-root "$PWD/mignet_ce/downstream/application/input_data" \
  --output-root "$PWD/mignet_ce/downstream/application/output" \
  --source-sample <T1_SAMPLE_ID> \
  --target-sample <T2_SAMPLE_ID>
```

---

## Plan 2

```bash
python -m mignet_ce.downstream.application.cli.run_seurat_cg \
  --prepared-root "$PWD/mignet_ce/downstream/application/output" \
  --output-root "$PWD/mignet_ce/downstream/application/output"
```

---

## Plan 3

```bash
python -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --prepared-root "$PWD/mignet_ce/downstream/application/output" \
  --output-root "$PWD/mignet_ce/downstream/application/output" \
  --k 40 \
  --device auto
```

---

## Plan 4

```bash
python -m mignet_ce.downstream.application.cli.run_biological_cluster \
  --prepared-root "$PWD/mignet_ce/downstream/application/output" \
  --output-root "$PWD/mignet_ce/downstream/application/output"
```

---

# 13. 测试计划

## 13.1 Plan 1

必须验证：

```text
只发现/准备 2 samples
exactly one source
exactly one target
CCI runner 只调用一次
GRN runner 只调用一次
CCI/GRN sample list 长度为 2
不出现 B1/C1/D1 loop
```

---

## 13.2 factory bridge

测试：

```text
真实运行不 capture stdout/stderr
错误 return code fail-fast
命令完整打印
mock runner 仍可用于 unit tests
```

---

## 13.3 common spot reference

验证：

```text
第一次 build
第二次直接 cache hit
Plan 2 K150/K40 使用同一路径
Plan 4 使用同一路径
PIJ_spot 数值未发生变化
```

---

## 13.4 Plan 2

验证：

```text
exactly one pair
K150 跑一次
K40 跑一次
Spot PIJ 不重复构造
K150/K40 的 macro CCI/GRN 正常
ΔEI / CQ 正常输出
```

---

## 13.5 Plan 3

验证：

```text
只启动一个 WYT training
--k 正确传入
summary K 与 CLI 一致
Keff 单独记录
无 B1/C1/D1 输出目录
```

---

## 13.6 Plan 4

验证：

```text
只分析唯一 source/target
直接加载 common PIJ
不重新 build spot dynamics
biological mapping 对齐
ΔEI / CQ 正常
```

---

# 14. 回归测试：与 v1 比较

选择与 v1 完全相同的一对 T1/T2。

必须比较：

```text
prepared H5AD shape
spot IDs
gene IDs
CCI 文件
GRN 文件
CCI/GRN 参数
```

如果 v1 与修复后 v2 使用同一核心 Data Factory 与同一参数：

```text
Plan 1 的计算规模应恢复到同一量级
```

允许因为：

```text
provenance
validation
output organization
```

出现少量额外时间，但不应再出现由 3 pair 调度造成的约 3× 工作量。

---

# 15. 性能验收

修复后执行：

```text
prepare_data
```

必须在开始时明确打印：

```text
Source sample: ...
Target sample: ...
Samples to process: 2
```

CCI 开始：

```text
Running spot CCI for 2 samples
```

GRN 开始：

```text
Running spot GRN for 2 samples
```

GRN 运行过程中必须可以看到底层实时进度。

禁止再次出现：

```text
Plan 1 started
→ 几小时完全没有任何可见输出
```

---

# 16. 不允许通过降低算法参数“修快”

本 Plan 的目标是修复调度，不是通过改变实验定义减少计算。

默认禁止为了加速擅自修改：

```text
GRN n_trees
GRN top_hvg
CCI 参数
Seurat 参数
NMF 参数
Optimal CG epoch
K
CQ 定义
```

如果后续需要 smoke test，可以另外显式提供 test 参数，但正式默认参数必须保持原算法配置。

---

# 17. 最终修复后的结构

```text
mignet_ce/downstream/application/
├── data/
│   └── Plan 1 adapter
├── common/
│   └── one authoritative spot reference
├── seurat_cg/
│   └── one pair → K150/K40
├── optimal_cg/
│   └── one pair → one WYT training
├── biological_cluster/
│   └── one pair → fixed biological CG
├── cli/
│   ├── prepare_data.py
│   ├── run_seurat_cg.py
│   ├── run_optimal_cg.py
│   └── run_biological_cluster.py
└── tests/
```

---

# 18. 最终验收标准

Plan 5 完成后必须满足：

1. `prepare_data` 一次只处理用户指定的一个 source 和一个 target；
2. 不再自动运行 B1/C1/D1 三对；
3. Plan 1 的 CCI / GRN 计算量恢复到 v1 的单 pair 量级；
4. GRN/CCI 长任务实时显示输出；
5. Plan 2 只处理这一对 T1/T2；
6. Plan 2 K150/K40 共用同一个 Spot PIJ；
7. Plan 3 只运行一次 Optimal CG；
8. Plan 3 的 K 可以通过 `--k` 指定；
9. Plan 4 只处理这一对 biological cluster maps；
10. Plan 4 复用 common Spot PIJ；
11. 不修改任何 CCI / GRN / Seurat / PIJ / EI / CQ / WYT 核心算法；
12. 所有修改只发生在 `mignet_ce/downstream/application/`。

最终原则：

```text
v1 的单 T1→T2 实验语义
+
v2 的模块化 application 架构
+
统一缓存 / 实时日志 / 清晰 provenance
```
