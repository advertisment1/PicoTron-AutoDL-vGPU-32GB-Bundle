# Picotron 2-GPU Ultra-Scale 实践（中文）

## 运行环境
- **同一台 Linux 云服务器上的 2 张 NVIDIA GPU**，不是两台各一张的服务器（跨机需要额外的网络与 rendezvous 配置）。
- VS Code Remote-SSH 打开云服务器上的仓库；Python 插件和 Python Debugger 插件在远程环境启用。
- 建议使用兼容的 Linux CUDA/PyTorch 环境。仓库 requirements.txt 中 torch==2.1.0、triton==2.1.0、flash-attn==2.5.0 与 transformers==4.47.0 的组合可能有兼容问题，先按 README/依赖核验，不保证一键安装。
- 两卡可见：`nvidia-smi -L`；`python -c "import torch; print(torch.__version__,torch.cuda.device_count())"`。

## 0. 不训练模型的三个真正源码实验
从项目根目录运行：
```bash
python -m pip install -e .
python -m pip install debugpy
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode groups
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode dp
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode tp
```
以上脚本调用**原 Picotron** ProcessGroupManager、DataParallelBucket、ColumnParallelLinear、RowParallelLinear，不是自己另写的玩具并行算法。结果保存在 `ultrascale_lab/results/`，通过 assert 验证 DP 梯度与 TP 输出的数值等价。

## 1. VS Code Debug
F5 选择 `Picotron 2GPU: groups (torchrun)`，在 `picotron/process_group_manager.py` 的 `self.grid = ...` 和 `dist.new_subgroups_by_enumeration(...)` 打断点；观察 `self.grid`、`self.dp_rank`、`self.dp_group_ids`。
F5 选择 `Picotron 2GPU: dp (torchrun)`，在 `picotron/data_parallel/data_parallel.py` 的 `_make_param_hook`、`_post_backward` 和 `picotron/data_parallel/bucket.py` 的 `dist.all_reduce` 相关位置打断点；观察 `param.grad`、`param.main_grad`、`bucket`。F5 选择 `Picotron 2GPU: tp (torchrun)`，在 `ColumnParallelLinear.forward`、`RowParallelLinear.forward` 和 `ReduceFromModelParallelRegion` 打断点；观察 `self.weight.shape`、`x.shape`、`output.shape`。

**分布式 Debug 警告：** 两个进程在集合通信前后停住时容易互相等待。优先用条件断点 `dist.get_rank() == 0`，不要在一个 Rank 的 NCCL collective 内长时间暂停；必要时重启实验。VS Code 子进程调试依赖 debugpy 版本/远程环境，不保证所有环境自动附加成功。若断点不命中，先使用命令行运行、日志结果定位，再用独立进程 attach。

## 2. 完整训练配置
```bash
python -m ultrascale_lab.make_configs --model HuggingFaceTB/SmolLM2-135M --steps 12
# 在确认数据集、HF 权限、显存、配置兼容后，再执行：
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py --config ultrascale_lab/configs/dp2.json
```
**重要**：`single.json` 只能用 `--nproc_per_node=1`，其余 `dp2/tp2/pp2/cp2` 用 2。配置默认 `gradient_accumulation_steps=4`，对 PP=2 的 1F1B 可用；但固定 `mbs=1` 时单卡与双卡 DP 的 global batch 不相等，严谨 benchmark 需手动调节 GAS 保持 token/step 一致。CP 可能需要额外依赖或 GPU 支持；先跑 DP/TP 的纯计算验证。

## 3. 课程路线
Day1 groups：解释四维 grid 与 Rank 映射，提交 JSON 和调试截图。
Day2 DP：解释 gradient hook、bucket、All-Reduce，证明双卡梯度等价。
Day3 TP：解释权重列/行切分与 All-Reduce，证明输出等价。
Day4 CP：调试 `apply_context_parallel`、`cp_communications`，观察 KV 交换。
Day5 PP：调试 `train_step_pipeline_1f1b` 与 `pp_communications`，绘制 micro-batch 时间线。
Day6 显存/吞吐：基线与四种配置，记录真实指标和瓶颈。
Day7 项目文档：对照 `INTERVIEW_STORY.md` 写结论、局限和面试问答。

## 4. 需要改进或核验的点
- 原 `train.py` 记录 `torch.cuda.memory_reserved()`，它不等于峰值实际 tensor allocated；报告应补 `max_memory_allocated()` / `max_memory_reserved()`。
- 原 `extract_metrics.py` 的 `process_file()` 在数据恰好 3 个点时，丢掉前三个后均值为空；正式跑至少 8-12 step，检查日志格式。
- `create_config.py` 会下载 HF 模型并清理同名实验目录，本课程提供独立的 `make_configs.py`，只生成 JSON、不删实验数据。
- 两张 GPU 不能同时设置 DP=2、TP=2；必须满足 DP*TP*PP*CP=2。
