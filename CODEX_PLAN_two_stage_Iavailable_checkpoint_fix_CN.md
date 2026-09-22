# CODEX PLAN：修复 two-stage 最优粗粒化中的低 I_available、状态塌缩与 checkpoint 选择错误

> 适用基线：当前正式代码 `code(20260922-030235).zip`  
> 对照来源：朋友版本 `shr_code(1).zip`  
> 目标方法：`maturity_cci_grn_two_stage`  
> 本计划要求：**先复现、再修复、再做单元测试与真实数据回归；不得通过修改 canonical PIJ 或随意调参掩盖问题。**

---

## 0. 任务目标与总体原则

当前正式代码已经集成了朋友开发的 maturity + CCI + GRN two-stage 最优粗粒化方法，但实际运行时出现了一个关键异常：

- 可以得到较高的 `ΔEI`；
- 但最终 learned coarse-graining 的 `I_available` 有时极低；
- `I_retained` 也可能接近 0；
- downstream 因此把结果判为 `low-signal`；
- 朋友原始代码/原始实验中没有这么严重的低 `I_available` 问题。

经过源码审查和真实数据测试，目前可以确认：**当前 canonical PIJ 不是主要根因**。真正的问题集中在 two-stage trainer 的以下几条逻辑链：

1. Stage 1 仅按 `ΔEI` 追求最优，没有足够的防塌缩约束；
2. `keff_min=12`、`checkpoint_keff_min=10` 被写成固定绝对值，与 multiscale 中 `K=10` 的正式配置冲突；
3. checkpoint 评估使用的是 `optimizer.step()` 之前的指标，但保存的是 `optimizer.step()` 之后的模型参数，产生 metric/state 错位；
4. Stage 2 如果没有 strict eligible checkpoint，会 fallback 到“最高 ΔEI”，容易重新选回 information-poor/collapsed 解；
5. trainer 与 downstream 使用两套不同的 low-signal threshold；
6. strict Stage 2 checkpoint 当前使用 `CQ` 作为第一排序指标，存在“比例很好、绝对信息很少”的偏好。

本次修改目标不是重新设计整套方法，而是：

> **修复已被实测确认的训练/选择逻辑错误，使 two-stage 方法只能从“非退化、信息充分”的 coarse-graining 中寻找高 ΔEI 和高动力学闭合解。**

---

# 1. 已确认的异常与实验依据

Codex 在修改前必须先理解这些现象，不能跳过这一节直接改代码。

## 1.1 当前 canonical PIJ 不应回滚

当前正式代码使用的是新的 canonical raw-KL NG PIJ（当前协议标识以仓库实际代码为准，例如 `canonical_ng_rawkl_v2`）。

真实 heart 11.5→12.5 Spot 数据测试中：

| 实现 | EI_micro |
|---|---:|
| 朋友 `shr_code` | 约 0.8585 bit |
| 当前正式代码 | 约 0.9472 bit |

两张真实 PIJ 的整体差异也很小：

- flattened correlation ≈ 0.993；
- Top-1 target overlap ≈ 80%；
- mean row JS ≈ 0.0024 bit。

更关键的是，把两套 PIJ 投影到同一个固定 Seurat K40 target mapping 后：

| PIJ | I_available |
|---|---:|
| friend PIJ | 约 0.5391 bit |
| current PIJ | 约 0.5775 bit |

因此：

> **当前 PIJ 本身可以提供 0.5 bit 以上的宏观可用未来信息。**

如果 learned `S_tp` 最终只得到 `I_available≈0.00x`，问题主要发生在 coarse-graining optimization / checkpoint selection，而不是 PIJ。

### 本次修复禁止事项

严禁为了修低 `I_available` 去修改：

- canonical PIJ 数学；
- NG 的 `alpha`；
- OT temperature / `tau`；
- Sinkhorn；
- CCI 特征构造；
- GRN 特征构造；
- NMF 数学；
- maturity 特征数学；
- `complete_combined.py` 中当前 production PIJ 协议。

如果 Codex 发现必须修改这些内容才能让测试通过，应停止并报告，而不是擅自修改。

---

## 1.2 Stage 1 可以选中“高 ΔEI、几乎无未来信息”的退化解

在当前正式代码、真实：

- heart；
- 11.5→12.5；
- input scale = `seurat_k40`；
- optimized `K=10`；
- seed = 42；

的诊断中，Stage 1 出现过如下 best-ΔEI 状态：

```text
ΔEI           ≈ 0.5774
I_available   ≈ 0.00252 bit
I_retained    ≈ 0.0000305 bit
Keff_t        ≈ 1.20
Keff_tp       ≈ 1.06
```

这说明当前 Stage 1 的 best checkpoint 可以是：

\[
\text{高 }\Delta EI
\quad+\quad
\text{严重 prototype / assignment collapse}
\quad+\quad
\text{几乎无动力学信息}.
\]

这不是随机噪声，而是目标函数/selection criterion 允许的退化解。

---

## 1.3 Stage 2 实际上能够恢复信息，但 final selection 会把恢复结果丢掉

在 seed = `20260809` 的真实诊断轨迹中，Stage 2 后期曾达到大约：

```text
I_available ≈ 0.531 bit
I_retained  ≈ 0.143 bit
```

说明：

> **Stage 2 本身并不是必然导致低 I_available。**

但是由于 strict checkpoint eligibility 没有通过，当前代码最后触发：

```text
stage2_highest_delta_ei_fallback_no_eligible_joint
```

最终又回到了高 ΔEI、低 retained-information 的状态，最终 `I_retained` 只有约：

```text
0.00225 bit
```

所以必须区分：

- “模型没学到”；
- “模型学到了，但 checkpoint 选错了”。

目前实测支持后者。

---

## 1.4 当前 K=10 配置与固定 Keff 阈值数学上冲突

当前 `WYTTwoStageDeltaEIConfig` 默认：

```python
keff_min = 12.0
checkpoint_keff_min = 10.0
```

而正式 multiscale runner 当前默认：

```python
DEFAULT_K_BY_SCALE = {
    "spot": [40],
    "seurat_k150": [40],
    "seurat_k40": [10],
}
```

因此 `seurat_k40 -> optimized K=10` 时：

### 训练 Keff floor

```text
Keff >= 12
```

数学上不可能，因为最大 `Keff <= K = 10`。

### checkpoint Keff floor

```text
Keff >= 10
```

要求几乎完美均匀使用全部 10 个宏状态，实际非常苛刻。

这会直接造成：

```text
Stage 2 一个 strict eligible checkpoint 都没有
        ↓
触发 fallback
        ↓
fallback 又按照最高 ΔEI
        ↓
重新选回 collapsed / low-information state
```

这是一个确定的配置逻辑错误，必须修复。

---

## 1.5 当前 checkpoint 存在 pre-step metric / post-step state 错位

当前 `two_stage_trainer.py` 的逻辑顺序是：

```text
forward
计算 delta_EI / I_available / I_retained / Keff / CQ
计算 loss
loss.backward()
optimizer.step()

根据 optimizer.step() 之前的指标判断 checkpoint
torch.save(optimizer.step() 之后的 model.state_dict())
```

因此：

\[
\text{checkpoint metadata}
\]

描述的是参数：

\[
\theta_e
\]

但保存进去的 model state 实际已经是：

\[
\theta_{e+1}.
\]

真实测试中曾出现：

```text
checkpoint 记录的 ΔEI ≈ +0.6083
重新载入该 checkpoint 后实际 ΔEI ≈ -0.028
```

这说明这不是小数误差，而是 checkpoint correctness bug。

**这项必须作为 P0 修复。**

---

# 2. 修改范围

## 2.1 允许修改的核心文件

优先限制修改在：

```text
wyt_deltaei_coarse_grain/two_stage_trainer.py
scripts/run_wyt_deltaei_coarse_grain.py
mignet_ce/downstream/analysis/dynamic_closure/analysis.py
```

允许新增一个无业务耦合的小型共享 helper：

```text
mignet_ce/information_thresholds.py
```

允许为了测试性新增：

```text
wyt_deltaei_coarse_grain/two_stage_selection.py
```

但只有在确实能让 checkpoint eligibility / ranking 变成纯函数并易于测试时才新增。

允许新增：

```text
tests/
```

中的单元测试与集成测试。

`scripts/run_multiscale_maturity_cci_grn.py` 只有在需要补充 resolved threshold/provenance 或参数透传时才修改。

---

## 2.2 禁止修改

本任务不得修改：

```text
mignet_ce/pij/...
wyt_deltaei_coarse_grain/complete_combined.py
mignet_ce/coarse_frontends/complete_combined_coarse_maturity_cci_grn.py
mignet_ce/coarse_frontends/maturity_cci_grn_two_stage.py
```

中的 PIJ、CCI、GRN、maturity 数学。

尤其禁止：

- 改 alpha；
- 改 tau；
- 改 NMF components / 默认 NMF 数学；
- 改 Sinkhorn；
- 改 EI 定义；
- 改 I_available / I_retained 数学定义；
- 为了让测试“变好看”直接降低 `ei_retain_ratio=0.85`；
- 直接固定 seed=20260809 并把它当作修复；
- 删除或绕过 Stage 2；
- 把 K=10 强行改回 K=40 来逃避 Keff bug。

---

# 3. P0-1：修复 checkpoint metric/state 错位

文件：

```text
wyt_deltaei_coarse_grain/two_stage_trainer.py
```

## 3.1 当前错误

当前代码在：

```python
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

之后才：

- 构造 `row`；
- 判断 `best_ei.pt`；
- 判断 `best_joint.pt`；
- 保存 checkpoint。

但 `row` 中的 tensor 是 optimizer step 之前 forward 得到的。

因此必须让：

```text
用于判断 checkpoint 的 metrics
```

与：

```text
实际保存的 model parameters
```

严格属于同一个 state。

---

## 3.2 推荐的最小改法

**不要在本任务中大规模重写 epoch 语义。**

保持现在“本 epoch 的 metrics 来自当前 forward state”的语义，只把：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(...)
optimizer.step()
```

移动到：

1. metrics row 构造；
2. checkpoint eligibility；
3. checkpoint save；
4. logging；

之后。

目标顺序：

```text
optimizer.zero_grad()

forward
↓
计算所有 loss / metric
↓
构造 row
↓
append metrics
↓
checkpoint selection
↓
如果需要，保存此刻的 model + optimizer state
↓
logging
↓
loss.backward()
↓
clip_grad_norm
↓
optimizer.step()
```

这样：

```text
checkpoint["delta_EI"]
checkpoint["I_available"]
checkpoint["I_retained"]
checkpoint["Keff_*"]
```

与 checkpoint 中保存的：

```text
encoder.state_dict()
macro_net.state_dict()
optimizer.state_dict()
```

对应同一个参数状态。

---

## 3.3 checkpoint 中必须增加审计字段

`best_ei.pt` 与 `best_joint.pt` 至少保存：

```text
epoch
stage
selection
delta_EI
I_available
I_retained
closure_quality
Keff_t
Keff_tp
signal_status
signal_threshold_bits
resolved_keff_min
resolved_checkpoint_keff_min
```

不要只保存部分字段。

---

## 3.4 必须增加 checkpoint reload consistency test

测试逻辑：

1. 跑一个很小的 two-stage synthetic / tiny real case；
2. 产生 checkpoint；
3. 保存时记录 metric；
4. 重新 load checkpoint；
5. 对同一 prepared input 重新 forward/evaluate；
6. 检查：

```python
abs(delta_EI_reloaded - delta_EI_recorded) < 1e-7
abs(I_available_reloaded - I_available_recorded) < 1e-7
abs(I_retained_reloaded - I_retained_recorded) < 1e-7
abs(Keff_t_reloaded - Keff_t_recorded) < 1e-7
abs(Keff_tp_reloaded - Keff_tp_recorded) < 1e-7
```

如果 GPU 浮点误差明显，可放宽到 `1e-6`，但不能使用 `1e-3` 这种掩盖逻辑错误的容差。

---

# 4. P0-2：把固定 Keff floor 改成 K-relative default

这是必须修改的第二项。

## 4.1 目标语义

当前 K=40 时原默认：

```text
keff_min = 12
checkpoint_keff_min = 10
```

分别等价于：

\[
12/40=0.30
\]

和：

\[
10/40=0.25.
\]

因此默认规则改成：

\[
Keff_{\text{train,min}}=0.30K
\]

\[
Keff_{\text{checkpoint,min}}=0.25K.
\]

于是：

| K | resolved keff_min | resolved checkpoint_keff_min |
|---:|---:|---:|
| 40 | 12.0 | 10.0 |
| 10 | 3.0 | 2.5 |

这有一个重要性质：

> **K=40 的历史默认行为完全不变，只修复 K 较小时的数学冲突。**

---

## 4.2 必须保留 CLI 显式 override

现有 CLI：

```text
--keff-min
--checkpoint-keff-min
```

必须继续支持。

推荐语义：

```text
如果用户显式传入 absolute value：
    使用该 absolute value
否则：
    使用 K-relative default
```

不要静默 clamp 用户显式值。

例如：

```bash
--k 10 --keff-min 12
```

应该直接：

```text
ValueError / SystemExit
```

并明确提示：

```text
explicit keff_min=12 exceeds K=10
```

而不是偷偷变成 10。

---

## 4.3 推荐 Config 结构

可以将：

```python
keff_min: float
checkpoint_keff_min: float
```

改为可选：

```python
keff_min: float | None = None
checkpoint_keff_min: float | None = None
```

增加内部 resolved helper，例如：

```python
def resolve_keff_floor(
    *,
    k: int,
    explicit_value: float | None,
    default_ratio: float,
    name: str,
) -> float:
    ...
```

常量：

```python
DEFAULT_KEFF_MIN_RATIO = 0.30
DEFAULT_CHECKPOINT_KEFF_MIN_RATIO = 0.25
```

最终训练期间统一使用：

```text
resolved_keff_min
resolved_checkpoint_keff_min
```

而不是在各处分散计算。

---

## 4.4 validate 必须检查

至少：

```text
0 < resolved_keff_min <= K
0 < resolved_checkpoint_keff_min <= K
resolved_checkpoint_keff_min <= resolved_keff_min
```

如果设计上不要求第三条，也必须说明理由并写测试。

---

# 5. P0-3：Stage 1 必须真正使用 Keff anti-collapse loss

当前 Stage 1 已经计算：

```python
keff, keff_t, keff_tp = keff_floor_loss(...)
```

但是 Stage 1 loss 里没有：

```python
lambda_keff * keff
```

Stage 2 才有。

这会导致 Stage 1 可以仅凭高 ΔEI 收敛到：

```text
Keff ≈ 1
```

的退化状态。

---

## 5.1 修改 Stage 1 loss

当前：

```python
loss = (
    lambda_align * align
    - lambda_ei * delta
    + lambda_var * var
    + lambda_local * local
    + lambda_sharp * sharp
    + lambda_dev * dev
    + lambda_proto * proto
    + lambda_dead_usage * usage
)
```

修改为：

```python
loss = (
    lambda_align * align
    - lambda_ei * delta
    + lambda_var * var
    + lambda_local * local
    + lambda_sharp * sharp
    + lambda_dev * dev
    + lambda_proto * proto
    + lambda_dead_usage * usage
    + lambda_keff * keff
)
```

不要额外引入 uniform-balance loss。

本方法需要的是：

> 防止有效宏状态数塌缩；

而不是：

> 强迫所有 macro state 完全均匀。

保留当前 `keff_floor_loss` 的设计即可。

---

## 5.2 metrics 中 Stage 1 的 `weighted_keff` 必须真实反映 loss

目前 `metrics.csv` 已经写：

```text
weighted_keff
```

修改后 Stage 1 该列不能再只是“计算了但没有进入 loss”的诊断值。

应保证：

```text
stage1 loss 实际包含 weighted_keff
```

并加入测试。

---

# 6. P1-1：Stage 1 best checkpoint 增加“非退化 eligibility gate”

当前 Stage 1 selection：

```python
if stage == "stage1" and row["delta_EI"] > best_delta:
    save best_ei.pt
```

这是导致：

```text
ΔEI=0.577
I_available=0.0025
Keff≈1
```

仍能成为 Stage 2 reference 的直接原因。

---

## 6.1 Stage 1 eligible 条件

Stage 1 checkpoint 至少同时要求：

### 条件 A：动力学信号不是 low-signal

\[
I_{\text{available}}
\ge
\tau_{\text{signal}}(K)
\]

### 条件 B：source 和 target 都没有 Keff collapse

\[
Keff_t \ge Keff_{\text{checkpoint,min}}
\]

\[
Keff_{tp} \ge Keff_{\text{checkpoint,min}}
\]

只有满足这些条件以后，才在 eligible candidates 中：

\[
\max \Delta EI.
\]

也就是：

\[
S^{*}_{Stage1}
=
\arg\max_{S\in\text{non-degenerate candidates}}
\Delta EI(S).
\]

不要改变为直接最大化 `I_available`。

Stage 1 的主目标仍然是 ΔEI，只是禁止退化解成为 reference。

---

## 6.2 如果 Stage 1 没有任何 eligible checkpoint

**禁止**自动退回 unconstrained highest-ΔEI。

正确行为：

1. `metrics.csv` 正常写出；
2. 输出清晰诊断；
3. fail-fast；
4. 不进入 Stage 2。

错误信息至少包含：

```text
best unconstrained ΔEI
对应 I_available
对应 I_retained
对应 Keff_t
对应 Keff_tp
signal threshold
resolved checkpoint Keff floor
```

这样以后才能判断：

- loss 权重是否真的有问题；
- 还是数据本身没有可行 coarse-graining。

不能再 silently 接受 information-poor reference。

---

# 7. P1-2：统一 trainer 与 downstream 的 low-signal threshold

当前存在两套定义。

## trainer

```python
low_signal_threshold_bits = 0.005
```

## downstream dynamic closure

```python
max(0.01, 0.01 * log2(K))
```

这会导致同一个 `I_available=0.03`：

```text
trainer: informative
downstream: low-signal
```

不允许继续存在。

---

## 7.1 建立唯一共享 helper

新增：

```text
mignet_ce/information_thresholds.py
```

建议只放纯函数：

```python
from __future__ import annotations
import math

def closure_signal_threshold_bits(k: int) -> float:
    if k < 2:
        raise ValueError(...)
    return max(0.01, 0.01 * math.log2(k))
```

---

## 7.2 trainer 使用方式

`WYTTwoStageDeltaEIConfig.low_signal_threshold_bits` 推荐改成：

```python
float | None = None
```

语义：

```text
None:
    使用 closure_signal_threshold_bits(K)

显式传值:
    尊重 CLI override
```

resolved threshold 必须在：

```text
train.log
metrics.csv
best_ei.pt
best_joint.pt
summary
```

中有可追溯记录。

---

## 7.3 downstream 使用方式

把：

```text
mignet_ce/downstream/analysis/dynamic_closure/analysis.py
```

中的两处：

```python
max(0.01, 0.01 * np.log2(...))
```

改成调用同一个：

```python
closure_signal_threshold_bits(K)
```

数学不能改变。

---

# 8. P1-3：Stage 2 strict eligibility 必须加入 I_available floor

当前 `joint_checkpoint_eligible()` 只检查：

```text
ΔEI floor
I_retained floor
Keff floor
```

但没有检查：

```text
I_available 是否仍然 informative
```

因此需要增加：

```python
available_information
signal_threshold_bits
```

参数。

strict eligibility 改为：

\[
\Delta EI
\ge
\rho_{EI}\Delta EI_{ref}
\]

且

\[
I_{\text{retained}}
\ge
\rho_R I_{\text{retained,ref}}
\]

且

\[
Keff_t,Keff_{tp}
\ge
Keff_{\text{checkpoint,min}}
\]

且

\[
I_{\text{available}}
\ge
\tau_{\text{signal}}(K).
\]

这样即使 CQ 很高，也不能让 low-signal candidate 进入 strict checkpoint ranking。

---

# 9. P1-4：改变 strict checkpoint ranking，避免 CQ-first

当前：

```python
score = (
    closure_quality,
    normalized_retained,
    -L_within_dynamics,
    mean_pairwise_inter_js,
    delta_EI,
)
```

Python tuple 是 lexicographic ranking，因此当前实际上优先：

```text
最大 CQ
```

但：

\[
CQ =
\frac{I_{\text{retained}}}{I_{\text{available}}}
\]

是比例指标。

例如：

```text
0.010 / 0.011
```

可能比：

```text
0.600 / 0.700
```

拥有更高 CQ，但前者几乎没有绝对信息。

虽然加入 I_available gate 后风险会降低，但 final strict score 仍建议调整。

---

## 9.1 推荐 strict ranking

在 candidates 已经通过：

- EI floor；
- retained floor；
- I_available signal floor；
- Keff floor；

之后，排序改成：

```python
score = (
    normalized_retained,
    closure_quality,
    -L_within_dynamics,
    mean_pairwise_inter_js,
    delta_EI,
)
```

其中：

```text
normalized_retained = I_retained / frozen_stage1_I_available_reference
```

因为 denominator 对同一个 run 是固定的，所以这等价于优先保留更多绝对动力学信息。

重要：

> 不要删除 CQ；只是把它从第一优先级降为第二优先级。

---

# 10. P1-5：彻底取消“highest ΔEI 作为最终 Stage 2 fallback”

这是当前最危险的 selection 行为之一。

当前逻辑：

```text
没有 strict eligible checkpoint
    ↓
best_stage2_fallback.pt = Stage 2 highest ΔEI
    ↓
复制成 best_joint.pt
```

这与 Stage 2 的设计目标矛盾。

Stage 2 本来就是为了在保留 EI 的前提下优化动力学闭合和 retained information。

如果最后又按 ΔEI 回退，就等于：

> Stage 2 训练了很多 epoch，最终还是回到最像 Stage 1 的 state。

---

## 10.1 新增 relaxed informative fallback

Stage 2 同时维护两个候选池：

### A. strict candidate

满足：

```text
EI floor
retained floor
I_available signal floor
Keff floor
```

按 strict score 排序。

### B. relaxed informative candidate

如果 strict candidate 一个都没有，允许 relaxation，但必须继续满足：

```text
I_available >= signal threshold
Keff_t >= checkpoint Keff floor
Keff_tp >= checkpoint Keff floor
ΔEI > 0
```

即：

> 可以放松“相对 Stage1 的 EI retain ratio / retained reference floor”，  
> 但不能放松“是否有真实动力学信息”和“是否已经塌缩”。

relaxed fallback 推荐排序：

```python
score = (
    I_retained,
    closure_quality,
    delta_EI,
    -L_within_dynamics,
    mean_pairwise_inter_js,
)
```

或使用 `normalized_retained` 作为第一项，两者在同一 run 中等价。

最终 checkpoint 标记：

```text
selection = "relaxed_informative_stage2_fallback"
```

summary 中：

```text
strict_checkpoint_found = false
fallback_used = true
fallback_type = "relaxed_informative"
```

---

## 10.2 如果连 relaxed informative candidate 都没有

不要再返回 highest ΔEI。

正确行为：

```text
NoValidCoarseGraining / RuntimeError
```

并写出 diagnostics。

允许保留一个：

```text
best_stage2_deltaei_diagnostic.pt
```

纯用于 debug，

但是它：

```text
绝对不能复制为 best_joint.pt
绝对不能成为 final arrays / paper result
```

---

# 11. 暂时不要修改 `ei_retain_ratio=0.85`

当前 strict EI floor：

\[
\Delta EI
\ge
0.85\Delta EI_{Stage1}
\]

可能最终需要做 sensitivity，但这次**不要一开始就调**。

原因：

目前已经确认存在更基础的逻辑错误：

1. Stage 1 collapse；
2. K=10 impossible Keff constraint；
3. checkpoint state mismatch；
4. bad highest-ΔEI fallback；
5. inconsistent low-signal criterion。

这些修完之前：

```text
0.85 是否过严
```

无法被公平评估。

因此 Codex 本轮必须：

```text
保持 ei_retain_ratio 默认 0.85 不变
```

修复后再通过真实 1500-epoch regression 观察 strict eligible rate。

如果修复后绝大多数数据仍然没有 strict candidate，再单独给出 sensitivity 报告：

```text
0.85
0.80
0.75
```

但**不要在本任务中自动改默认值。**

---

# 12. Stage 2 frozen reference 逻辑暂时保留，但必须建立在“合格 Stage1 reference”上

目前 Stage 2 使用：

```text
EI_reference
I_available_reference
I_retained_reference
```

冻结自 `best_ei.pt`。

本任务先不要重新设计这套 two-stage 理论。

修复 Stage 1 eligibility 后：

```text
best_ei.pt
```

必须已经满足：

- informative；
- non-collapsed；
- best ΔEI among feasible states。

这样 Stage 2 reference 至少不再是：

```text
I_available≈0.002
Keff≈1
```

的退化 reference。

之后再评估 frozen-reference 设计是否需要进一步改变。

---

# 13. summary / provenance 必须补充的字段

最终 summary 至少新增或保证存在：

```text
K
resolved_keff_min
resolved_checkpoint_keff_min
signal_threshold_bits

stage1_best_epoch
stage1_best_delta_EI
stage1_reference_I_available
stage1_reference_I_retained
stage1_reference_Keff_t
stage1_reference_Keff_tp
stage1_reference_signal_status

strict_checkpoint_found
best_joint_selection
fallback_used
fallback_type

final_I_available
final_I_retained
final_closure_quality
final_Keff_t
final_Keff_tp

checkpoint_metric_state_consistent
```

现有字段名称如果已经存在，不要重复造第二套名字。

优先保留 backward-compatible names，并只补缺失字段。

---

# 14. `metrics.csv` 增加 selection audit 字段

每个 epoch 建议增加：

```text
stage1_eligible
stage2_strict_eligible
stage2_relaxed_eligible
signal_threshold_bits
resolved_keff_min
resolved_checkpoint_keff_min
```

如果不希望每行重复 resolved constants，至少在 summary/provenance 中必须有。

对于 eligibility，建议每行写布尔值。

这样后续可以直接画：

```text
ΔEI
I_available
I_retained
CQ
Keff
eligibility
```

随 epoch 的轨迹。

---

# 15. 需要新增的纯函数/测试友好结构

为了避免 selection 逻辑继续散落在训练 loop 中，推荐把以下逻辑做成纯函数。

可以放：

```text
wyt_deltaei_coarse_grain/two_stage_selection.py
```

建议函数：

```python
resolve_keff_floor(...)
stage1_checkpoint_eligible(...)
joint_checkpoint_eligible(...)
relaxed_stage2_checkpoint_eligible(...)
strict_joint_score(...)
relaxed_fallback_score(...)
```

这些函数：

- 不访问文件；
- 不依赖 torch model；
- 输入普通 float；
- 输出 bool / tuple；
- 可以直接 unit test。

不要为了重构而移动训练数学，只提取 checkpoint policy。

---

# 16. 单元测试要求

当前正式 zip 中没有看到成熟的 tests 目录，因此本次必须新增。

建议：

```text
tests/
├── test_two_stage_keff_resolution.py
├── test_two_stage_checkpoint_policy.py
├── test_two_stage_checkpoint_consistency.py
├── test_closure_signal_threshold.py
└── test_two_stage_stage1_keff.py
```

---

## 16.1 `test_two_stage_keff_resolution.py`

必须检查：

### K=40 默认

```text
resolved_keff_min == 12.0
resolved_checkpoint_keff_min == 10.0
```

### K=10 默认

```text
resolved_keff_min == 3.0
resolved_checkpoint_keff_min == 2.5
```

### explicit override

```text
K=10
explicit keff_min=4
=> 4
```

### invalid explicit

```text
K=10
explicit keff_min=12
=> raise
```

---

## 16.2 `test_closure_signal_threshold.py`

验证：

\[
\tau(K)=\max(0.01,0.01\log_2K)
\]

例如：

```text
K=10 -> 约 0.0332192809
K=40 -> 约 0.0532192809
```

并检查：

```text
trainer
downstream information_closure_budget
downstream information_closure_budget_from_observed
```

使用同一个 helper。

---

## 16.3 `test_two_stage_checkpoint_policy.py`

构造 fake rows。

### Case A：高 ΔEI，但低信息

```text
deltaEI=0.60
I_available=0.002
I_retained=0.001
Keff=1.1
```

必须：

```text
stage1_eligible == False
stage2_strict_eligible == False
stage2_relaxed_eligible == False
```

### Case B：稍低 ΔEI，但 informative

```text
deltaEI=0.52
I_available=0.20
I_retained=0.08
Keff=5
```

K=10 时必须可进入 eligible pool。

### Case C：比较 strict ranking

Candidate 1：

```text
CQ=0.90
I_retained=0.03
```

Candidate 2：

```text
CQ=0.75
I_retained=0.15
```

如果两者都满足 hard gates，

**Candidate 2 必须优先**，

因为 retained information 更丰富。

---

## 16.4 `test_two_stage_checkpoint_consistency.py`

这是 P0 必测。

产生实际 checkpoint 后：

```text
recorded metric
```

与 reload 后 recompute 必须一致。

特别验证：

```text
delta_EI
I_available
I_retained
Keff_t
Keff_tp
```

不能再出现：

```text
record +0.608
reload -0.028
```

---

## 16.5 `test_two_stage_stage1_keff.py`

验证 Stage 1 loss 确实包含：

```text
lambda_keff * L_keff
```

不要只测试 metrics 字段存在。

建议构造一个明显 collapsed assignment，使：

```text
L_keff > 0
```

比较：

```text
lambda_keff=0
lambda_keff=1
```

下 Stage1 total loss 的差值应等于对应 `L_keff`（允许数值容差）。

---

# 17. 真实数据回归测试：必须做，不允许只跑 unit test

---

## 17.1 回归 A：Seurat K40 → optimized K=10

这是最快、也是当前最容易暴露 bug 的配置。

数据：

```text
organ = heart
time pair = 11.5 -> 12.5
input scale = seurat_k40
optimized K = 10
epochs = 1500
```

至少跑两个 seed：

```text
42
20260809
```

其中：

- 42 是当前正式默认；
- 20260809 是朋友正式实验中使用过的 seed，可用于复现对照。

---

## 17.2 修改前基线参考

当前旧逻辑曾出现：

```text
ΔEI           ≈ 0.5774
I_available   ≈ 0.00252
I_retained    ≈ 0.0000305
Keff_t        ≈ 1.20
Keff_tp       ≈ 1.06
```

这些数值仅用于说明旧问题。

**不得把它们硬编码进代码。**

---

## 17.3 修改后验收

最终 checkpoint 必须满足：

### 信息性

\[
I_{\text{available}}
\ge
\tau(10)
\approx0.0332193.
\]

### 非塌缩

\[
Keff_t\ge2.5
\]

\[
Keff_{tp}\ge2.5.
\]

### 因果涌现

\[
\Delta EI > 0.
\]

### selection

final：

```text
best_joint_selection
```

不得是：

```text
stage2_highest_delta_ei_fallback_no_eligible_joint
```

因为这个 fallback 应被移除。

### checkpoint consistency

reload 后所有关键 metrics 与 summary 一致。

---

## 17.4 已有临时修复实验可作为 sanity reference

此前只在临时副本中做过初步修复，出现过：

```text
ΔEI          ≈ 0.5102
I_available  ≈ 0.2099
I_retained   ≈ 0.0392
Keff_t       ≈ 5.04
Keff_tp      ≈ 3.31
```

该结果证明：

> 提高 I_available 并不要求牺牲掉全部 ΔEI。

这些也只是 sanity reference，不是必须精确复现的 golden value。

---

# 18. 真实数据回归 B：Spot → optimized K=40

在 K10 回归全部通过后，再跑：

```text
organ = heart
time pair = 11.5 -> 12.5
input scale = spot
optimized K = 40
epochs = 1500
```

至少 seed：

```text
42
```

如果资源允许再跑：

```text
20260809
```

---

## 18.1 Spot K40 最低验收

signal threshold：

\[
\tau(40)
=
0.01\log_2(40)
\approx0.0532193.
\]

最终：

```text
I_available >= 0.0532193
Keff_t >= 10
Keff_tp >= 10
ΔEI > 0
```

如果 strict candidate 不存在，但 relaxed informative fallback 存在：

- 可以输出 relaxed result；
- 必须明确标记；
- 不允许伪装成 strict。

如果 relaxed 都不存在：

- run 应失败；
- 不得输出一个 collapsed final result。

---

# 19. 对朋友结果的比较方式

朋友 `shr_code` 中 heart 11.5→12.5 曾有大约：

```text
EI_micro     ≈ 0.8598
EI_macro     ≈ 2.3136
ΔEI          ≈ 1.4538
I_available  ≈ 0.7237
I_retained   ≈ 0.6589
CQ           ≈ 0.9105
```

本任务**不要求 current 代码精确复现这些数值**，因为：

- canonical PIJ 已更新；
- 当前项目 preprocessing/协议已有变化；
- seed 和 frontend 环境也可能不同。

朋友结果的作用是：

> 证明 two-stage coarse-graining 理论上可以得到“高 ΔEI + 高 absolute information + 高 CQ”，而不是只能得到 low-information 高 CQ。

因此不要为了“追上 0.7237”进行参数拟合。

---

# 20. 对 multiscale runner 的兼容性检查

文件：

```text
scripts/run_multiscale_maturity_cci_grn.py
```

当前：

```python
DEFAULT_K_BY_SCALE = {
    "spot": [40],
    "seurat_k150": [40],
    "seurat_k40": [10],
}
```

修改后必须验证三个 scale 都能正确 resolve：

```text
spot K40          -> Keff floor 12 / checkpoint 10
seurat_k150 K40   -> Keff floor 12 / checkpoint 10
seurat_k40 K10    -> Keff floor 3 / checkpoint 2.5
```

不要在 runner 里复制第二套 Keff 计算逻辑。

应该由：

```text
WYTTwoStageDeltaEIConfig / shared resolver
```

统一计算。

runner 只负责传 K。

---

# 21. 不要顺手修改 DEFAULT_METHOD

当前：

```python
DEFAULT_METHOD = "complete_combined_coarse_maturity_cci_grn"
```

而不是 two-stage。

这确实是一个容易误跑 legacy single-stage 的配置陷阱，但它不是本次低 `I_available` bug 的根因。

因此：

> 本次修复不要顺手把 multiscale `DEFAULT_METHOD` 改成 two-stage。

除非上层调用明确要求，否则避免把 bugfix 和行为默认值迁移混在一个 commit 中。

可以在最终报告里单独提醒：

```text
正式 two-stage run 必须显式：
--method maturity_cci_grn_two_stage
```

---

# 22. 保持输出兼容

不要删除或重命名现有主要 artifact：

```text
best_ei.pt
best_joint.pt
metrics.csv
train.log
assignment_t.npy / csv
assignment_tp.npy / csv
macro_pij
summary
strict posthoc evaluation
```

可以新增：

```text
best_stage2_deltaei_diagnostic.pt
```

但只能诊断使用。

如果新增 relaxed fallback checkpoint，可内部保存：

```text
best_relaxed_joint.pt
```

最终仍然统一：

```text
best_joint.pt
```

作为 downstream 使用的正式 checkpoint。

---

# 23. 最终 summary 必须能够回答以下问题

每个 run 不看 train.log，只看 summary 就应该知道：

1. Stage 1 reference 是哪一 epoch？
2. 它的 `ΔEI / I_available / I_retained / Keff` 是多少？
3. signal threshold 是多少？
4. resolved Keff thresholds 是多少？
5. Stage 2 是否找到 strict checkpoint？
6. 如果没找到，是否使用 relaxed informative fallback？
7. final selection 的理由是什么？
8. final checkpoint reload 后 metrics 是否与选择时一致？
9. 最终是否 informative？
10. 最终是否存在 prototype collapse？

---

# 24. 开发执行顺序

Codex 必须按照以下顺序做，不要一次改一大坨后再测试。

## Phase A：基线审计

先不改代码：

- 确认上述文件与函数仍与本 plan 一致；
- 如果最新代码已经有部分修复，记录；
- 跑已有 tests（如果有）；
- 建立修改前 diagnostic。

---

## Phase B：只修 checkpoint state mismatch

只修改 optimizer/checkpoint 顺序。

然后：

```text
unit test checkpoint consistency
```

必须先通过。

没有通过不要继续。

---

## Phase C：实现 K-relative Keff

加入：

```text
default 0.30K
checkpoint default 0.25K
explicit override validation
```

跑 Keff unit tests。

---

## Phase D：Stage 1 加 Keff loss + eligibility

加入：

```text
lambda_keff * keff
```

和 Stage1 informative/noncollapsed gate。

跑 K10 短程 diagnostic，确认不再轻易掉到：

```text
Keff≈1
I_available≈0
```

---

## Phase E：统一 low-signal threshold

新增 shared helper。

trainer/downstream 共用。

跑 threshold tests。

---

## Phase F：重写 Stage 2 selection/fallback policy

实现：

```text
strict informative joint
relaxed informative fallback
no valid -> fail
```

删除 final highest-ΔEI fallback。

改变 strict score 为 retained-first。

跑 pure policy tests。

---

## Phase G：K10 full 1500 回归

seed：

```text
42
20260809
```

全部通过才继续。

---

## Phase H：Spot K40 回归

至少 seed 42。

检查：

```text
I_available
I_retained
CQ
ΔEI
Keff
selection
```

---

## Phase I：全项目兼容检查

至少：

```bash
python -m compileall ...
pytest -q
```

并检查：

- paper assets 读取 summary 是否受影响；
- downstream dynamic closure 是否还能读取 assignment；
- multiscale aggregate 是否还能读取 summary；
- GRN perturbation baseline 构造 `WYTTwoStageDeltaEIConfig` 时是否适配新的 optional Keff defaults；
- `mignet_ce/downstream/analysis/preparation.py` 中对 config 的构造是否需要同步。

---

# 25. Codex 必须检查的调用点

不要只改 trainer。

用 grep 检查：

```text
WYTTwoStageDeltaEIConfig
keff_min
checkpoint_keff_min
low_signal_threshold_bits
joint_checkpoint_eligible
best_stage2_fallback
best_joint_selection
best_joint_fallback_used
```

至少检查：

```text
scripts/run_wyt_deltaei_coarse_grain.py
scripts/run_multiscale_maturity_cci_grn.py
mignet_ce/downstream/analysis/preparation.py
mignet_ce/downstream/analysis/grn_perturbation/baseline.py
mignet_ce/downstream/analysis/dynamic_closure/analysis.py
```

防止 config 类型从 float 改成 Optional 后其他调用点崩掉。

---

# 26. 最终验收标准

只有全部满足以下条件才算完成。

## 代码正确性

- [ ] checkpoint metric/state 不再错位；
- [ ] reload checkpoint 后关键 metrics 与记录一致；
- [ ] Stage 1 loss 确实包含 Keff floor；
- [ ] K10 不再使用不可能的 `keff_min=12`；
- [ ] explicit impossible Keff override 会 fail-fast；
- [ ] trainer/downstream low-signal threshold 一致；
- [ ] Stage1 low-information/collapsed state 不可成为 `best_ei.pt`；
- [ ] Stage2 low-signal state 不可成为 strict `best_joint.pt`；
- [ ] final fallback 不再使用 highest ΔEI；
- [ ] CQ 不再作为 strict ranking 的第一优先级。

## 方法不被偷改

- [ ] canonical PIJ 数学无改动；
- [ ] CCI/GRN/maturity 数学无改动；
- [ ] `ei_retain_ratio` 默认仍为 0.85；
- [ ] Stage1/Stage2 epoch ratio 默认仍为 0.45；
- [ ] K grid 不被修改；
- [ ] seed 默认不通过“改成朋友 seed”来解决问题。

## K10 真实回归

- [ ] `I_available >= tau(10)`；
- [ ] `Keff_t >= 2.5`；
- [ ] `Keff_tp >= 2.5`；
- [ ] `ΔEI > 0`；
- [ ] final selection 不是 highest-ΔEI fallback；
- [ ] seed 42 与 seed 20260809 均完成并输出比较表。

## Spot K40 回归

- [ ] `I_available >= tau(40)`；
- [ ] `Keff_t >= 10`；
- [ ] `Keff_tp >= 10`；
- [ ] `ΔEI > 0`；
- [ ] checkpoint reload consistency 通过。

---

# 27. 最终要求 Codex 提交的报告

代码修改完成后，不要只回复“done”。

必须给出一份中文报告，包含：

## A. 修改文件清单

逐文件列出：

```text
文件
修改内容
为什么改
是否改变数学方法
```

---

## B. Bug → Fix 对照表

例如：

| Bug | Root cause | Fix | Test |
|---|---|---|---|
| checkpoint 数值与 reload 不一致 | pre-step metric + post-step state | save before optimizer.step | consistency test |
| K10 永远达不到 Keff=12 | absolute default | 0.30K / 0.25K | K10/K40 unit |
| Stage1 高 ΔEI 低信息 | ΔEI-only + no Keff | Keff loss + eligibility | real K10 |
| Stage2 恢复后又被丢掉 | highest-ΔEI fallback | informative relaxed fallback | selection test |

---

## C. 修改前后真实结果表

至少：

### seurat_k40 → K10, heart 11.5→12.5

```text
seed=42
seed=20260809
```

列：

```text
version
seed
best_epoch
selection
ΔEI
I_available
I_retained
CQ
Keff_t
Keff_tp
signal_status
```

---

## D. Spot K40 验证结果

同样列：

```text
ΔEI
I_available
I_retained
CQ
Keff
selection
```

---

## E. 未解决事项

如果仍存在：

- strict checkpoint 很少；
- 0.85 EI floor 过严；
- retained/CQ tradeoff；
- seed sensitivity；

只记录，不要在本任务里继续偷偷调参。

下一轮再单独做 sensitivity。

---

# 28. 本任务的核心原则

最终实现必须满足下面这句话：

> **Stage 1 仍然以 ΔEI 为主目标，但只允许从非塌缩、具有足够可用未来信息的 coarse-graining 中选择 reference；Stage 2 在保持因果涌现的前提下优先保留更多动力学信息并提高闭合，不允许因为 strict constraint 未满足就退回一个只有高 ΔEI、却几乎没有未来信息的解。**

本次不是要把 two-stage 改成“最大化 I_available”。

也不是要把 CQ 删除。

而是防止如下错误解被系统性选成正式最优粗粒化：

\[
\Delta EI \text{ 很高},
\qquad
I_{\text{available}}\approx0,
\qquad
I_{\text{retained}}\approx0,
\qquad
Keff\approx1.
\]

这样的解即使 ΔEI 高，也不应该作为动力学闭合 two-stage 方法的正式输出。

---

# 29. Codex 开始开发前的最后检查

开始写代码前，请先回答并记录：

1. 当前 `two_stage_trainer.py` 是否仍在 `optimizer.step()` 后保存 pre-step metrics 对应的 checkpoint？
2. Stage1 loss 是否仍没有 `lambda_keff * keff`？
3. 当前 K10 默认是否仍解析成 `keff_min=12 / checkpoint=10`？
4. Stage1 是否仍然只按最高 ΔEI 保存 `best_ei.pt`？
5. Stage2 是否仍然存在 `stage2_highest_delta_ei_fallback_no_eligible_joint`？
6. strict score 是否仍然 CQ-first？
7. trainer 是否仍使用固定 `0.005 bit` low-signal threshold？
8. downstream 是否仍使用 `max(0.01, 0.01 log2 K)`？

如果最新代码与本 plan 有出入：

> 以“修复同一根因”为准适配，不要为了机械匹配行号把已经修好的地方改坏。

完成修改后，再用第 26 节的验收标准逐项打勾。
