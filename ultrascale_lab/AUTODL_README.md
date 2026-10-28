# AutoDL RTX 2080 Ti x2: execution guide

## Security
Reset the AutoDL SSH password disclosed in chat. Do not put credentials into project files or command history.

## Supported first pass
This package preserves original Picotron code and adds standalone probes for original process groups, DP bucket synchronization and TP row/column linear modules. RTX 2080 Ti (SM75) does not support native BF16 and should not be assumed compatible with FlashAttention-2. Original Picotron training configs with FLASH_ATTEN=1 should NOT be launched unchanged.

## Run
On the AutoDL instance, upload/unzip this archive and open a terminal in picotron-main:

```bash
python -m pip install --no-deps -e .
python -m pip install numpy
bash ultrascale_lab/run_autodl.sh
```

Do not install `requirements.txt` blindly: it pins torch 2.1.0, Triton 2.1.0 and flash-attn 2.5.0 while the supplied AutoDL image reports torch 2.8.0, CUDA 12.8, Python 3.12. The environment may require additional compatible packages if imports fail.

Results: `ultrascale_lab/results/{groups,dp,tp}_rank{0,1}.json` and `results/logs/*.log`.

## Status and limits
- Offline static compile checks are possible without GPUs.
- GPU correctness probes require running this script on the rented instance.
- Full PP/CP and SmolLM2 training are not certified for RTX 2080 Ti in this package.
- Do not claim measured performance until the generated logs have been checked.

## Debug
Use VS Code Remote-SSH to connect to the AutoDL instance after changing the password. Run one rank with `debugpy` only after confirming the non-debug multi-process run; pausing one rank at a collective can cause timeout. Use conditional breakpoints or attach to both ranks.
