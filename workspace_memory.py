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
        # learned base memory (Parameter) and dynamic per-sequence state (Buffer)
        init = torch.randn(slots, d) * 0.01
        self.mem = nn.Parameter(init.clone())
        self.register_buffer('state', init.clone())
        # factorized projections
        self.q1 = nn.Linear(d, rank, bias=False)
        self.k1 = nn.Linear(d, rank, bias=False)
        self.v1 = nn.Linear(d, rank, bias=False)
        self.o1 = nn.Linear(rank, d, bias=False)
        self.controller = nn.Linear(d, 2, bias=False)  # read/write gating logits
        self.ln = nn.LayerNorm(d)

    def reset_state(self):
        """Reset dynamic state from learned base memory for per-sequence fairness."""
        with torch.no_grad():
            self.state.copy_(self.mem.data)

    def forward(self, x):  # x (B,T,d)
        B, T, D = x.shape
        summary = x.mean(dim=1)  # (B,d)
        gate = torch.softmax(self.controller(summary), dim=-1)  # [read, write]
        # Write: blend top-k slots with summary into dynamic state
        with torch.no_grad():
            norms = self.state.norm(dim=-1)
            topk = norms.topk(self.active, largest=False).indices
        write_vec = gate[:, 1].unsqueeze(-1) * summary  # (B,d)
        nupd = min(topk.numel(), B)
        if nupd > 0:
            self.state.data[topk[:nupd]] = 0.9 * self.state.data[topk[:nupd]] + 0.1 * write_vec[:nupd]
        # Read: attend all slots from dynamic state
        m = self.state.unsqueeze(0).expand(B, self.slots, D)
        q = self.q1(self.ln(summary)).unsqueeze(1)
        k = self.k1(self.ln(m))
        v = self.v1(self.ln(m))
        att = torch.softmax((q * k).sum(-1) / (k.size(-1) ** 0.5), dim=-1)  # (B,slots)
        read = self.o1((att.unsqueeze(-1) * v).sum(1))  # (B,d)
        read = gate[:, 0].unsqueeze(-1) * read
        # Broadcast to tokens (simple additive)
        return x + read.unsqueeze(1)
