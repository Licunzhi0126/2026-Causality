# CCI 扰动与矩形转移矩阵的对称性实验

默认时间对为小鼠心脏 spot 层级 E11.5 → E12.5。首轮选 CCI 10%、50%、90% 和对称性共同预算 50%，共计划最多 43 个正式训练条件，包括共享 P0、30 个 CCI 条件和 12 个对称性扰动条件。存在性分析读取 P0，默认计算两种观测划分和 200 组大小匹配的随机划分。模块直接调用正式 frontend、训练器与公共指标。

## 运行

从服务器代码根目录执行，`--developmental-root` 指向包含 `spot/heart_11.5_features.csv` 和 `spot/heart_12.5_features.csv` 的目录。两份表提供 `unit_id` 和 `pseudotime`。CCI、H5AD、GRN 和 Seurat 映射使用数据工厂目录结构。

```bash
cd '/home/jovyan/work/2026 Causality'
python -u -m mignet_ce.downstream.post_perbutation \
  --experiment all --organ heart --layer spot \
  --source-time 11.5 --target-time 12.5 \
  --data-root /home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory \
  --developmental-root /path/to/developmental_features \
  --output-root output/post_perbutation/heart_11.5_to_12.5 \
  --devices cuda:0 cuda:1 --workers-per-gpu 3 --max-workers-per-gpu 6 \
  --gpu-budget-gib 18 --cpu-workers 32
```

- 加 `--dry-run` 可检查所有输入、矩阵和 ID 对齐、CCI 选区重叠与计划训练数。预检在 CPU 上运行，结果输出到终端。
- `--experiment cci`、`--experiment symmetry-existence` 和 `--experiment symmetry-perturbation` 可独立启动、续跑和导出。三个入口使用同一个 `--output-root`，各自写入 `cci/`、`symmetry_existence/`、`symmetry_perturbation/`，并按输入及训练配置复用 `baseline/` 里的同一份 P0。`all` 把三个队列合并运行，输出写到根目录。
- `--cci-topologies P1 --cci-strengths 10` 和 `--symmetry-levels 20` 可以缩小验证范围。完整网格用 `--cci-strengths 10 20 30 40 50 60 70 80 90 --symmetry-levels 20 50 80`，最多 127 个训练条件。
- 加 `--resume` 继续同一输出目录。校验输入哈希、配置、条件及已完成产物后，重试未完成条件。修改训练或扰动参数时使用新的输出目录。
- `--experiment report --output-root <目录>` 可合并各 pipeline 已保存的结果并重出表图。

## 多进程

训练从每卡三个进程开始。训练子进程绑定单卡，在完成条件后记录 CUDA 预留峰值。调度器结合 `nvidia-smi` 的整卡占用和进程峰值逐步开放更多位置，默认每卡最多六个。`--gpu-budget-gib 18` 是派发目标，计算时包含其它进程的整卡占用并留 1 GiB 余量。同一用户的多个 pipeline 实例通过本机临时目录里按 GPU UUID 划分的锁共享预算。`gpu_usage.json` 保留整卡采样峰值和最终并发上限。

在 128 核服务器上默认 32 个 CPU 工作进程，可用 `--cpu-workers` 调整。P0 和 CCI 的正式特征提取、KLOT 及输入准备在 CPU 池完成，GPU 进程读取准备好的数组后执行正式训练。CPU 池并行构造全局、局部和弱对称性的扰动端点，再校验请求的预算档。存在性的随机划分按批次分给 GPU 和 CPU。同一批次使用固定种子，结果按原顺序合并。CPU 和训练子进程的 BLAS 线程上限为 1。

终端与 `progress.json` 按组显示完成量。P0、CCI、对称性存在性、对称性扰动分别计数。失败与不可行的条件另计，续跑时已完成的条件或随机划分批次直接计入进度。

显存不足时，对应进程退出以释放显存。调度器等待一张卡空闲，再独占该卡重试一次，另一张卡继续执行其他条件。每个尝试保留单独的 `worker.log` 和结果文件。正式训练不满足 `best_joint.pt` 的选择条件时，日志和原有诊断 checkpoint 会随该次尝试保留。通过校验的条件由主进程写入总表。

## 实验口径

- CCI 在 P0 最强的 10% 有向非自环边上发现 P1–P5 选区，默认使用 40% 节点和 40% 候选边。九档强度是选定区域内嵌套的选边比例。正负条件共享边集合，在完整 T1 CCI 上分别翻倍和移除相应边。T2 使用原始输入。拓扑指标对完整 CCI 计算，模块度固定使用 P0 的分区，加权聚集同时报告固定 P0 权重尺度的版本。
- 全局对称性比较两端置换后的转移矩阵，存在性同时测量成对置换及循环置换。局部、弱对称性分别比较目标簇内概率的排序和总量。存在性至少报告每个原划分配套的随机划分、置换覆盖率、随机置换对照、微观 EI 和熵。
- 对称性扰动始终从 P0 的矩形 KLOT 矩阵开始。每类以实际搬移概率质量为单位，用四组已找到的扰动幅度上限中的最小值作为共同预算，再取 20%、50%、80%。`plan.json` 记录候选端点、实际强度、边缘误差和每个条件的方向校验。研究者可从计划与条件表查看不可行的条件。每个可行条件重新训练两阶段模型，训练前用固定硬划分检查干预方向，训练后用重新学习的硬划分检查三类对称性。
- 每个条件从同一个最佳 checkpoint 对应的导出矩阵获取 (P,S_t,S_{t+1},Q_{\mathrm{trained}})，从同一份 (P,S_t,S_{t+1}) 推导 (Q_{\mathrm{induced}})。两种宏观 EI 都减去同一个微观 EI。正式 CQ 使用 induced 信息预算，Closure JS 比较 (PS_{t+1}) 与 (S_tQ_{\mathrm{trained}})。

## 留档

`run_manifest.json` 记录输入及配置，`conditions.json` 与 `conditions.csv` 记录条件与结果，`plans/` 保存选区、可行预算和存在性分批结果。普通条件在 `runs/<条件>/attempt-<编号>/training/` 导出 Pij、软分配、训练 Q 和诱导 Q。共享 P0 在 `baseline/<输入指纹>/attempt-<编号>/training/`。每个实验的结果表会输出 PNG、PDF 和 CSV，文件名为 `cci_results_table`、`symmetry_existence_table` 和 `symmetry_perturbation_table`。

默认配置的训练参数为 `K=40`、1500 轮、训练种子 42，正式特征采用 NMF、Laplacian HKS 和 `alpha_cci=0.20`。
