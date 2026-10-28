# 面试项目故事：从 Ultra-Scale Playbook 到 Picotron 2-GPU 验证

## 问题与目标
在仅有两张 GPU 的约束下，解释和验证 DP/TP/PP/CP 的正确性、内存和吞吐取舍；不是声称做过大集群扩展。

## 实测环境（本次真实运行）
- 实例：AutoDL vGPU-32GB×2 = **2 × NVIDIA RTX 4080 SUPER**（compute capability 8.9 / Ada），32 GB 显存，驱动 595.71.05。
- 软件：Ubuntu 22.04，Python 3.12.3，torch 2.8.0+cu128，NCCL 2.27.3+cuda12.9，triton 3.4.0，flash-attn 2.8.3.post2（官方预编译 wheel），transformers 4.47.0，datasets 2.21.0。
- 模型：`HuggingFaceTB/SmolLM2-135M` 结构（hidden 576；`make_configs` 继承 base_config 的 16 头 / 4 KV 头覆盖，实际 **161.2M 参数**），30 层，`FLASH_ATTEN=1`（bf16 + flash-attn）。
- 数据：`roneneldan/TinyStories`，`seq_length=1024`，`micro_batch_size=2`，`gradient_accumulation_steps` 按配置调节使 **global batch = 8192 tokens/step** 一致（single GAS4 / dp2 GAS2 / tp2 GAS4 / pp2 GAS4），15 步，取 step 2–15 稳态平均。

## 实验方法
1. 对 Picotron 原始 ProcessGroupManager 进行 Rank/通信组调试（groups 探针）。
2. 使用原始 DataParallelBucket 的梯度与等效全局批次单进程参考计算比较（dp 探针）。
3. 使用原始 ColumnParallelLinear/RowParallelLinear 对比完整线性层结果（tp 探针）。
4. 运行 DP=2/TP=2/PP=2/CP=2 完整训练，PP=2 走 1F1B，CP=2 走 ring attention。
5. 固定模型、序列长度、有效 global batch tokens，对比每秒 token、显存、Loss；本次受实例时长限制只跑了 15 步（未做 3 次重复）。

## 实测结果表（数据来源：`results/vgpu4080/bench30L/BENCH_REPORT.md`）

| 配置 | GPU | tokens/s/GPU | step time(ms) | peak reserved(GB) | nvidia-smi峰值(GB) | Loss(15步) | 正确性 |
|---|---|---:|---:|---:|---:|---:|---|
| 基线 single | 1 | 11,293 | 726 | 3.98 | 5.35 | 6.31 | 前向/反向正常 |
| DP=2 | 2 | 10,392 | 394 | 4.23 | 4.87 | 6.32 | 梯度等价 1.19e-07 |
| TP=2 | 2 | 4,854 | 844 | 2.92 | 4.01 | 6.72 | 输出等价 4.77e-07 |
| PP=2 | 2 | 7,515 | 545 | 2.32 | 3.08 | 6.38 | 1F1B 跑通 |
| CP=2 | 2 | 3,671 | 1,116 | 2.68 | 3.46 | 6.41 | ring-attention 等价（见下） |

注：`peak reserved` 来自 `torch.cuda.memory_reserved()`（`train.py` 记录值）；「nvidia-smi峰值」来自训练期间 `nvidia-smi -lms 200` 采样的 `memory.used` 最大值。**`max_memory_allocated()` 未测**（需改 `train.py`，本次保持原版零改动）。

## 关键发现（真实数字支撑）

1. **DP=2 近线性扩展**：总吞吐 20.78K tokens/s ≈ 单卡 1.84×，每 GPU 效率 92%，只有约 8% 被梯度 All-Reduce 吃掉。
2. **TP=2 收益为负**：每 GPU 只有单卡的 43%（4.85K vs 11.29K）。hidden 576 / 16 头的小模型，每层每 micro-batch 两次 All-Reduce 的通信时间压过切分省下的计算——「小模型上 TP 不划算」的标准现象，属预期。
3. **PP=2 居中**（67%）：1F1B 有 bubble，通信量比 TP 少。
4. **MFU 全部只有 ~1%**（0.44%–1.34%）：161M 模型对 4080 SUPER 是 latency-bound，算力远未饱和。这些数字只反映「各并行策略在此模型规模下的相对开销」，**不能当作 4080S 能力上限**。
5. **CP=2 吞吐最低但等价性通过**：本版 CP 的 ring attention 是纯 PyTorch matmul + KV 环传递（非 flash kernel），慢是预期；其价值在超长序列的显存。**等价性判据**：cp2 的 Loss 曲线（11.00 → 6.41）与 single（10.98 → 6.31）几乎重合，说明切序列 + ring attention 数值正确。
6. **2080 Ti（SM75）跑不动的根因是架构不是显存**：flash-attn 2.x 要求 SM80+。换成 SM89 后，装上官方预编译 wheel，原版代码零改动即可运行。

## 面试问答

**Q：为什么 TP=2 反而更慢？这是 bug 吗？**
不是。TP 每次前向/反向都要在每层做两次 All-Reduce，通信量随层数线性增长；当单层计算量（hidden 576）小于通信时间时，切分省下的算力抵不过新增通信。这正是「通信/计算比」决定并行策略有效性的例子——大模型（hidden 大）上 TP 才会回正。

**Q：你的 MFU 只有 1%，说明什么？**
说明测试负载没有喂饱 GPU，不代表卡不行。MFU 低是因为模型太小、序列和 batch 不够大，算力被 kernel 启动和通信延迟主导。要证明卡的算力，得把计算密度（模型规模 × batch × seq）提上去。

**Q：DP 的 92% 扩展效率是怎么来的？**
DP 只在反向结束做一次梯度 All-Reduce（`DataParallelBucket` 把梯度打包成一个 bucket 同步），通信量 = 参数量，与层数无关、与序列长度无关，所以小模型上 DP 的固定开销最小，扩展效率最接近线性。

**Q：你说「验证正确性先于性能」，具体怎么做的？**
先跑三个探针：groups 验证 rank→四维网格映射；dp 验证双卡 All-Reduce 后梯度与单进程参考在 1.19e-07 内一致；tp 验证列/行切分 + All-Reduce 输出与全量线性层在 4.77e-07 内一致。确认数值等价后，才去测吞吐，避免「用错误结果换速度」。

**Q：2 卡能推断 8/64/512 卡的 scaling 吗？**
不能。2 卡只有 DP×TP×PP×CP 各一种二等分，验证的是机制正确性，不是 scaling law。更大规模下 All-Reduce 的 ring/tree 拓扑、PP 的 bubble 占比、CP 的 KV 环传递都会变化，需要更大集群才能给出 scaling efficiency。

## 必须诚实标注的范围
- Picotron 本版有 DP/TP/PP/CP，**不能宣称**它实现了 ZeRO-1/2/3、Sequence Parallel 或 Expert Parallel；ZeRO/SP/EP 是 Playbook 的理论比较。
- 本次只在「微测试 + 15 步吞吐」层面，未做完整收敛、未做 3 次重复、未测 `max_memory_allocated`。
- `tp2` 的 Loss（6.72）与其它配置（~6.3）略有差异，属 15 步短跑下的随机扰动，不构成收敛结论。

## 未完成项
- 无。DP/TP/PP/CP 四种并行 + 三个探针 + 吞吐对比 + 面试文档已全部完成。
