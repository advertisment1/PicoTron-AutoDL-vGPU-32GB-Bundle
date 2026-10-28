"""Create Picotron configs for 2 GPU; uses existing base_config, no downloads."""
import argparse,copy,json
from pathlib import Path
base=json.loads(Path('template/base_config.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M');p.add_argument('--steps',type=int,default=12)
a=p.parse_args()
for name,dp,tp,pp,cp in [('single',1,1,1,1),('dp2',2,1,1,1),('tp2',1,2,1,1),('pp2',1,1,2,1),('cp2',1,1,1,2)]:
 c=copy.deepcopy(base);c['distributed'].update(dp_size=dp,tp_size=tp,pp_size=pp,cp_size=cp,pp_engine='1f1b',use_cpu=False,backend='nccl')
 c['model'].update(name=a.model,num_hidden_layers=4,use_fused_adam=False)
 c['training'].update(total_train_steps=a.steps,seq_length=128,micro_batch_size=1,gradient_accumulation_steps=4, max_tokens=None)
 c['dataset'].update(num_workers=0,num_proc=1)
 c['checkpoint'].update(save_dir=f'ultrascale_lab/results/checkpoints/{name}',save_frequency=100000)
 c['logging'].update(use_wandb=False,run_name=name)
 c['environment'].update(FLASH_ATTEN='1',HF_TOKEN=None)
 out=Path('ultrascale_lab/configs')/f'{name}.json';out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(c,indent=2),encoding='utf-8');print(out)
print('CAUTION: model config and dataset downloads still required by original train.py; adjust model and batch before real runs.')
