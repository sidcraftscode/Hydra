"""Workspace memory placeholder implementation.
Provides simple slot read/write operations for integration tests.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkspaceMemory(nn.Module):
    def __init__(self, d: int, slots: int = 64, active: int = 32, rank: int = 128):
        super().__init__()
        self.d = d
        self.slots = slots
        self.active = active
        self.mem = nn.Parameter(torch.randn(slots,d)*0.01)
        # factorized projections
        self.q1 = nn.Linear(d, rank, bias=False)
        self.k1 = nn.Linear(d, rank, bias=False)
        self.v1 = nn.Linear(d, rank, bias=False)
        self.o1 = nn.Linear(rank, d, bias=False)
        self.controller = nn.Linear(d, 2, bias=False)  # read/write gating logits
        self.ln = nn.LayerNorm(d)
    def forward(self, x):  # x (B,T,d)
        B,T,D = x.shape
        summary = x.mean(dim=1)  # (B,d)
        gate = torch.softmax(self.controller(summary), dim=-1)  # read, write weights
        # simple write: blend top-k slots with summary
        with torch.no_grad():
            # choose active slots by small norm heuristic
            norms = self.mem.norm(dim=-1)
            topk = norms.topk(self.active, largest=False).indices  # least used
        write_vec = gate[:,1].unsqueeze(-1)*summary  # (B,d)
        self.mem.data[topk[:min(len(topk),B)]] = 0.9*self.mem.data[topk[:min(len(topk),B)]] + 0.1*write_vec[:min(len(topk),B)]
        # read: attend all slots
        m = self.mem.unsqueeze(0).expand(B,self.slots,D)
        q = self.q1(self.ln(summary)).unsqueeze(1)
        k = self.k1(self.ln(m))
        v = self.v1(self.ln(m))
        att = torch.softmax((q*k).sum(-1)/ (k.size(-1)**0.5), dim=-1)  # (B,slots)
        read = self.o1((att.unsqueeze(-1)*v).sum(1))  # (B,d)
        read = gate[:,0].unsqueeze(-1)*read
        # broadcast to tokens (simple additive)
        return x + read.unsqueeze(1)
