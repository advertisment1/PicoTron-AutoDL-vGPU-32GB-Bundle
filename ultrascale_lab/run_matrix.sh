#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
python -m ultrascale_lab.make_configs --steps 12
for mode in groups dp tp; do
  torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode "$mode"
done
# The training runs are separate: review config, GPU capacity and model downloads first.
# torchrun --standalone --nproc_per_node=2 train.py --config ultrascale_lab/configs/dp2.json
