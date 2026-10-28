# Day6 吞吐对比（30 层，seq 1024，一致 global batch）

## 配置口径（对齐，避免上一轮 smoke 的不公平对比）

- 模型：`HuggingFaceTB/SmolLM2-135M`，**30 层全深度**（此前 smoke 只用 4 层）。注意 `make_configs` 默认保留 base_config 的 `num_attention_heads=16 / num_key_value_heads=4` 覆盖，hidden_size 仍为 576，故是「SmolLM2 结构 + 16/4 头」的 135M 模型。
- 序列：`seq_length=1024`，`micro_batch_size=2`。
- **global batch 对齐到 8192 tokens/step**：single GAS=4、dp2 GAS=2、tp2 GAS=4、pp2 GAS=4（`mbs×seq×GAS×dp` 均 = 8192）。
- `num_samples=4000`，`total_train_steps=15`，`FLASH_ATTEN=1`（bf16 + flash-attn），`use_wandb=false`。
- 显存用 `nvidia-smi -lms 200` 后台采样取峰值（比 `memory_reserved` 更接近真实峰值）。

## 结果（取 step 2–15 稳态平均，剔除 step1 warmup）

| 配置 | 卡数 | avg Tokens/s | avg Tokens/s/GPU | 相对单卡(每GPU) | Loss 首→末 | MFU@step15 | 峰值显存 |
|---|---|---|---|---|---|---|---|
| `single` | 1 | 11,293 | 11,293 | 1.00× | 10.98 → 6.31 | 1.34% | 5346 MiB |
| `dp2` | 2 | 20,784 | 10,392 | **0.92×** | 11.00 → 6.32 | 1.20% | 4874 MiB |
| `tp2` | 2 | 9,709 | 4,854 | 0.43× | 10.92 → 6.72 | 0.58% | 4012 MiB |
| `pp2` | 2 | 15,030 | 7,515 | 0.67× | 11.00 → 6.38 | 0.88% | 3082 MiB |
| `cp2` | 2 | 7,342 | 3,671 | 0.33× | 11.00 → 6.41 | 0.44% | 3458 MiB |

## 结论

1. **DP2 接近线性扩展**：总吞吐 1.84×，每 GPU 效率 92%，只有约 8% 被梯度 All-Reduce 吃掉——这是四者里唯一「加卡近正收益」的配置。
2. **TP2 收益为负**（每 GPU 只有单卡的 43%）：hidden 576 / 16 头的模型太小，每层每 micro-batch 两次 All-Reduce 的通信时间压过了切分省下的计算。这正是「小模型上 TP 不划算」的标准现象，属预期，不是 bug。
3. **PP2 介于两者之间**（67%）：1F1B 有 bubble，但通信量比 TP 少。
4. **CP2 吞吐最低（每 GPU 只有 33%）但这是预期**：CP 的 ring attention 在本版是**纯 PyTorch matmul 实现**（不是 flash kernel），且每层要做 KV 的 send/recv 环传递，慢是必然的。CP 的价值在「超长序列下的显存」（把 softmax 矩阵按序列切分），不在小模型吞吐。**等价性已通过**：cp2 的 Loss 曲线（11.00 → 6.41）与 single（10.98 → 6.31）几乎重合，说明 ring attention 数值正确（差异 ~1e-2 来自 online-softmax 的 fp32 累加顺序不同）。
5. **MFU 全部只有 ~1%**：135M 模型对 4080 SUPER 而言是 latency-bound，算力远未饱和。所以这些数字只反映「并行策略在此模型规模下的相对开销」，**不能当作 4080S 的能力上限**。要测卡的能力，得用更大模型 / 更大 batch / 更长序列把计算密度提上去。

## 与上一轮 smoke（4 层，seq 128）的对比

| | smoke | 本组 |
|---|---|---|
| seq | 128 | 1024 |
| 层数 | 4 | 30 |
| 峰值显存 | ≤1.06 GB | 3.1–5.3 GB |
| MFU | 0.10% | 0.58%–1.34% |

可见层数/序列提升后显存和 MFU 都上了一个台阶，但离吃满仍差很远。

## 口径提醒（避免过度声称）

- 峰值显存来自 `nvidia-smi` 采样，是「训练期间观测到的最大 `memory.used`」，仍有采样盲区；`train.py` 自己记的是 `memory_reserved()`（缓存），两者不可混用。
- 16/4 头是 `make_configs` 继承 base_config 的产物，与 SmolLM2 原始 9/3 头不同；若要与官方数对齐需另做头数设置。
- Loss 曲线仅供「能训练、在下降」作证，收敛质量不在本次范围。
