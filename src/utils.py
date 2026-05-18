import torch
import random
import numpy as np

def apply_stats(X: torch.Tensor, st) -> torch.Tensor:
    mu = torch.tensor(st["mu"], dtype=torch.float32)
    sd = torch.tensor(st["sigma"], dtype=torch.float32)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    # broadcasting: (B,T,C,L) -> subtract per-channel mu and divide by sigma
    return (X - mu[None, None, :, None]) / (sd[None, None, :, None] + 1e-8)

def set_seeds(seed=60):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
