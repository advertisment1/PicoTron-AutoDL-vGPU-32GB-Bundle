"""Two-GPU instrumentation of the actual Picotron process-group and TP modules.
Run with torchrun --standalone --nproc_per_node=2 -m ultrascale_lab.distributed_probe --mode groups|dp|tp
"""
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import picotron.process_group_manager as pgm
from picotron.process_group_manager import setup_process_group_manager

def log(name, payload):
    rank = dist.get_rank()
    root = Path('ultrascale_lab/results')
    root.mkdir(parents=True,exist_ok=True)
    with (root / f'{name}_rank{rank}.json').open('w',encoding='utf-8') as f:
        json.dump(payload,f,indent=2,ensure_ascii=False)
    print(f'[rank {rank}] {name}: {payload}',flush=True)

def groups():
    p=pgm.process_group_manager
    log('groups',dict(rank=p.global_rank,local_rank=p.local_rank,grid=p.grid.tolist(),
        coordinate=[p.dp_rank,p.pp_rank,p.cp_rank,p.tp_rank],
        dp_group_ids=p.dp_group_ids,tp_group_ids=p.tp_group_ids,
        cp_group_ids=p.cp_group_ids,pp_group_ids=p.pp_group_ids))

def dp(device):
    # Same initial weights, different per-rank examples; compare all-reduced gradient to serial global batch.
    from picotron.data_parallel.data_parallel import DataParallelBucket
    rank=dist.get_rank()
    torch.manual_seed(2026)
    base=torch.nn.Linear(4,2,bias=False).to(device)
    original=base.weight.detach().clone()
    wrapped=DataParallelBucket(base,bucket_cap_mb=1)
    x=torch.arange(8,dtype=torch.float32,device=device).reshape(2,4)/10+rank
    target=torch.ones((2,2),device=device)*rank
    loss=F.mse_loss(wrapped(x),target)
    loss.backward()
    # Callback synchronizes grads before returning from backward.
    grad=base.weight.grad.detach().clone()
    grads=[torch.empty_like(grad) for _ in range(dist.get_world_size())]
    dist.all_gather(grads,grad)
    xs=[torch.empty_like(x) for _ in range(dist.get_world_size())]
    ys=[torch.empty_like(target) for _ in range(dist.get_world_size())]
    dist.all_gather(xs,x);dist.all_gather(ys,target)
    reference=torch.nn.Linear(4,2,bias=False).to(device)
    with torch.no_grad():reference.weight.copy_(original)
    F.mse_loss(reference(torch.cat(xs)),torch.cat(ys)).backward()
    error=(grad-reference.weight.grad).abs().max().item()
    assert error<1e-5, f'DP mismatch {error}'
    log('dp',dict(local_loss=loss.item(),synced_grad=grad.cpu().tolist(),
                  serial_reference_error=error,all_ranks_equal=torch.allclose(grads[0],grads[1]),pass_test=True))

def tp(device):
    # Exact Picotron row/column modules, with deterministic manually partitioned weights.
    from picotron.tensor_parallel.tensor_parallel import ColumnParallelLinear,RowParallelLinear
    rank=dist.get_rank()
    full_up=torch.arange(32,dtype=torch.float32,device=device).reshape(8,4)/40
    full_down=torch.arange(24,dtype=torch.float32,device=device).reshape(3,8)/30
    x=torch.arange(8,dtype=torch.float32,device=device).reshape(2,4)/8
    col=ColumnParallelLinear(4,8,bias=False).to(device)
    row=RowParallelLinear(8,3,bias=False).to(device)
    with torch.no_grad():
        col.weight.copy_(full_up.chunk(2,dim=0)[rank]);row.weight.copy_(full_down.chunk(2,dim=1)[rank])
    local=col(x)
    output=row(local)
    reference=F.linear(F.linear(x,full_up),full_down)
    error=(output-reference).abs().max().item()
    assert error<1e-5, f'TP mismatch {error}'
    log('tp',dict(input_shape=list(x.shape),column_weight_shape=list(col.weight.shape),
                  column_output_shape=list(local.shape),row_weight_shape=list(row.weight.shape),
                  final_output_shape=list(output.shape),max_abs_error=error,pass_test=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['groups','dp','tp'],required=True)
    ap.add_argument('--cpu',action='store_true',help='Offline Gloo validation without CUDA')
    args=ap.parse_args()
    assert torch.cuda.device_count()>=2, 'This course targets a single machine with two visible NVIDIA GPUs.'
    local_rank=int(os.environ['LOCAL_RANK'])
    if not args.cpu: torch.cuda.set_device(local_rank)
    device=torch.device('cpu' if args.cpu else 'cuda', None if args.cpu else local_rank)
    dist.init_process_group('gloo' if args.cpu else 'nccl')
    # group setup must be called by every rank in identical order
    tp_size=2 if args.mode=='tp' else 1
    dp_size=2 if args.mode!='tp' else 1
    setup_process_group_manager(tp_size=tp_size,cp_size=1,pp_size=1,dp_size=dp_size)
    try:
        if args.mode=='groups':groups()
        elif args.mode=='dp':dp(device)
        elif args.mode=='tp':tp(device)
        dist.barrier()
    finally:
        dist.destroy_process_group()
if __name__=='__main__':main()
