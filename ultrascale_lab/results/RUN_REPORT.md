# AutoDL RTX 2080 Ti x2 实跑记录

- 主机：`region-42.seetacloud.com:12779`（autodl-container-2cce11be52-6486e4f6，Ubuntu 22.04.5）
- GPU：2 × NVIDIA GeForce RTX 2080 Ti 11GB，驱动 580.105.08
- 环境：Python 3.12.3（/root/miniconda3）、torch 2.8.0+cu128、NCCL 2.27.3+cuda12.9、triton 3.4.0
- 代码位置：`/root/picotron-main`（`pip install --no-deps -e .` 成功）

## 执行过的命令

```bash
python -m pip install --no-deps -e .
bash ultrascale_lab/run_autodl.sh      # groups / dp / tp 三个探针
python -m ultrascale_lab.make_configs --model HuggingFaceTB/SmolLM2-135M --steps 12
```

## 实测结果（全部 pass）

| 模式 | 关键输出 | 判定 |
|---|---|---|
| groups | rank0 coordinate `[0,0,0,0]`、rank1 `[1,0,0,0]`；两 rank `dp_group_ids=[0,1]`；tp/cp/pp 各为单元素组。grid 形状为 4 维 `[[[[0]]],[[[1]]]]` | 四维 DPxTPxPPxCP 网格正确，DP=2 |
| dp | 两 rank `synced_grad` 逐位相同；`all_ranks_equal=true`；与串行参考 `serial_reference_error=1.19e-07` | 双卡梯度等价（浮点误差级） |
| tp | `column_weight [4,4]`→`column_output [2,4]`，`row_weight [3,4]`→`final_output [2,3]`；`max_abs_error=4.77e-07` | 列/行切分 + All-Reduce 输出等价 |

原始 JSON：`groups_rank{0,1}.json`、`dp_rank{0,1}.json`、`tp_rank{0,1}.json`；日志：`logs/{groups,dp,tp}.log`、`run_all.log`。

## 全量训练的阻塞点（未启动）

1. `picotron/model.py` 第 7-9 行**无条件** import `flash_attn`，未安装则 `import picotron.model` 直接 ModuleNotFoundError；因此 `FLASH_ATTEN=0` 也无法绕开。
2. 环境里 `transformers`、`datasets`、`accelerate` 均缺失，`flash_attn` 缺失；flash-attn 2.x 要求 SM80+，2080 Ti 是 SM75。
3. `make_configs` 生成的配置默认 `FLASH_ATTEN='1'`，与 2080 Ti 不兼容（AUTODL_README 已明确不要原样启动）。

结论：本次只完成官方支持的「微测试」路径，未产生任何训练吞吐/显存数据，不能据此声称性能。
