# vGPU-32GB ×2（RTX 4080 SUPER）实跑报告

## 一、结论先说

**原版 Picotron 代码零改动，SmolLM2-135M 在两张 vGPU-32GB 上完整训练跑通**（`single` / `dp2` / `tp2` / `pp2` 四条路径全部成功），官方三个探针也全部 pass。

之前 2080 Ti 跑不动的根因是**架构**（SM75 不支持 flash-attn 2.x），不是显存。换成 SM89 后这个限制消失。

## 二、环境

| 项目 | 值 |
|---|---|
| 实例 | `autodl-container-e7a74b9082-45bb5db4`，Ubuntu 22.04.5 |
| GPU | 2 × NVIDIA GeForce RTX 4080 SUPER，**compute capability 8.9（Ada / SM89）**，32760 MiB |
| 驱动 / CUDA | 595.71.05 / CUDA 12.8 |
| Python / torch | 3.12.3 / 2.8.0+cu128，NCCL 2.27.3，triton 3.4.0 |
| transformers / datasets | **4.47.0 / 2.21.0**（从镜像默认的 5.20.2 / 5.1.0 降回仓库同时期版本） |
| flash-attn | **2.8.3.post2**（预编译 wheel，非源码编译） |

## 三、关键工程要点：flash-attn 怎么装上的

**源码编译这条路走不通（也别浪费时间）**：`pip install flash-attn --no-build-isolation` 在这台机器上卡在 `Building wheel for flash-attn (setup.py): started` 之后不动——进程还活着、有 65 个线程、打开了 `/dev/nvidiactl` 和 `/dev/nvidia-uvm`，但**不产生任何 nvcc 子进程**，即 setup.py 阶段就挂住，与 CUDA 无关（单独测 `torch.randn(...).cuda()` 完全正常）。

**有效路径：直接装官方预编译 wheel。**

```bash
# 1) 先确认 torch 的 C++ ABI 与 python 版本
python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"   # -> True
# 2) 从 Dao-AILab/flash-attention release 取匹配的 wheel（v2.8.3.post2）
#    flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps /root/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

装完三个原始依赖符号全部可用（这是"不改代码"成立的前提）：

```
flash_attn_func          OK   (flash_attn.flash_attn_interface)
apply_rotary_emb         OK   (flash_attn.layers.rotary)
layer_norm_fn            OK   (flash_attn.ops.triton.layer_norm)   # 2.8.3 仍然保留，没被移除
```

注意两点坑：`pip install fa.whl` 会报 `fa.whl is not a valid wheel filename`，**必须保留完整 wheel 文件名**；`pkill -f 'pip install flash-attn'` 会把自己那条 bash 命令行也匹配掉从而自杀，要写成 `'pip install flash[-]attn'`。

## 四、实测数据

### 4.1 官方探针（`bash ultrascale_lab/run_autodl.sh`，exit 0）

| 模式 | 结果 | 判定 |
|---|---|---|
| groups | DP=2 网格，rank0 `[0,0,0,0]` / rank1 `[1,0,0,0]`，dp_group_ids 均为 `[0,1]` | pass |
| dp | 两 rank 梯度逐位相同，`all_ranks_equal=True`，`serial_reference_error=1.19e-07` | pass |
| tp | `max_abs_error=4.77e-07` | pass |

数值与 2080 Ti 上完全一致（同为 FP32 探针，不含硬件相关路径）。

### 4.2 真实训练（SmolLM2-135M，4 层，seq_len=128，mbs=1，GAS=4，TinyStories 2000 样本）

| 配置 | 卡数 | tokens/step | 步数 | Loss 首 → 末 | 稳态 tokens/s | tokens/s/GPU | memory_reserved |
|---|---|---|---|---|---|---|---|
| `single` | 1 | 512 | 12 | 11.00 → **8.16** | ≈2.4K | 2.40K | 0.79 GB |
| `dp2` | 2 | 1024 | 12 | 11.00 → **8.18** | ≈4.44K | 2.22K | 1.06 GB |
| `tp2` | 2 | 512 | 3（冒烟） | 10.98 → 10.52 | ≈2.06K | 1.03K | 0.70 GB |
| `pp2` | 2 | 512 | 3（冒烟） | 10.95 → 10.59 | ≈3.44K | 1.72K | 0.63 GB |

解读：

- **DP2 相对单卡加速 ≈1.85×**（4.44K / 2.40K），接近线性；按每 GPU 吞吐算 2.22K vs 2.40K，只掉 7.5%，符合 All-Reduce 有固定开销的预期。
- **TP2 比单卡还慢**（2.06K < 2.40K）。4 层 / hidden 576 这类小模型，每次 TP 都需要两次 All-Reduce，通信占比过高，属于正常的反例，不是 bug。
- **PP2 居中**（3.44K），1F1B 有 bubble，但通信量比 TP 少。
- 显存全部在 1.1GB 以内，32GB 卡完全是过剩——**这次选卡买到的是架构，不是显存**。

## 五、必须写进报告的口径说明（避免过度声称）

1. **MFU 0.10% 不是性能结论**。`num_hidden_layers=4`、`seq_length=128`、`micro_batch_size=1` 是为了在 2080 Ti 上冒烟而设的极小配置，算力完全没喂满。要做真正的吞吐/MFU 对比，需要把层数、seq_len、batch 提到能压满 GPU 的量级，并用一致的 tokens/step。
2. **global batch 不相等**：`single` 是 512 tokens/step，`dp2` 是 1024，因为 `make_configs` 固定 mbs=1、GAS=4。严谨 benchmark 必须调 GAS 让两边 tokens/step 一致（START_HERE 第 2 节也点了这条）。
3. **显存口径**：`train.py` 记录的是 `torch.cuda.memory_reserved()`，不等于峰值 `max_memory_allocated()`。上表只能当缓存占用参考。
4. `tp2` / `pp2` 只跑了 3 步，其 Loss 不能和 12 步的对比。
5. 日志里 `Checkpoint has 272 layers but model only has 38 layers` 是**预期告警**：只取 SmolLM2 的前 4 层权重，其余层丢弃。

## 六、代码改动状态

- 远端 `/root/picotron-main/picotron/model.py`：**原版，md5 `da13ae274456c952b1d211b9132edeca`，一行未改**。
- 本地 `picotron/model.py`：**已还原为同一原版**（md5 相同）。此前为 2080 Ti 写的「可选 flash_attn import」补丁已撤销——它在 SM89 + flash-attn 环境下本来就是 no-op，留着只会让课程要截图的原始代码不干净。若将来还要回 SM75 机器，补丁只需 6 处：`try/except` 包住三个 flash_attn import + `HAS_FLASH_ATTN`/`use_flash_attn()`、`flash_attention()` 守卫、`TritonRMSNorm.forward` 守卫、`Attention.forward` 的两个分支条件、两处 `RMSNorm` 选择。
- 唯一被改的是生成物 `ultrascale_lab/configs/*.json`：`num_samples=2000`、`num_proc=8`、`total_train_steps=12`（tp2/pp2 为 3）。

