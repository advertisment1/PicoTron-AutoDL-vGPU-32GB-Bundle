import torch

def reset_peak():
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

def snapshot():
    if not torch.cuda.is_available(): return {}
    torch.cuda.synchronize()
    return {'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'current_allocated_gib':torch.cuda.memory_allocated()/2**30}
