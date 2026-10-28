#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
mkdir -p ultrascale_lab/results/logs
python - <<'PY'
import torch
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU count:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
 print('GPU',i,torch.cuda.get_device_name(i),'bf16:',torch.cuda.is_bf16_supported())
assert torch.cuda.device_count()>=2,'Two GPUs must be visible'
PY
for mode in groups dp tp; do
 echo "==== $mode ===="
 torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode "$mode" 2>&1 | tee "ultrascale_lab/results/logs/${mode}.log"
done
printf '\nCompleted validated micro-tests. Full LLM training requires a separate compatibility check for RTX 2080 Ti.\n'
