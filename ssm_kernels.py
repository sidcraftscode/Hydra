"""Placeholder for real selective scan / Mamba-style SSM kernels.
Currently provides a torch implementation approximating selective scan; to be replaced with fused kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveScanSSM(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(d, 4*d, bias=False)
        self.decay = nn.Parameter(torch.ones(d))
        self.out_ln = nn.LayerNorm(d)
    def forward(self, x):
        B,T,D = x.shape
        delta,Bt,Ct,Ut = self.proj(x).chunk(4,dim=-1)
        alpha = torch.exp(-F.softplus(delta)*F.softplus(self.decay))
        s = torch.zeros(B,D,device=x.device,dtype=x.dtype)
        outs=[]
        for t in range(T):
            y = Ct[:,t]*s
            outs.append(y)
            s = alpha[:,t]*s + Bt[:,t]*Ut[:,t]
        y = torch.stack(outs,dim=1)
        return self.out_ln(y)

class FusedSelectiveScanSSM(SelectiveScanSSM):
    """Alias placeholder for a future CUDA fused version."""
    pass
