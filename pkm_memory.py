"""Product-Key Memory placeholder (toy)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PKMMemory(nn.Module):
    def __init__(self, d: int, n1: int = 64, n2: int = 64, key_dim: int = 64, value_dim: int = 128, topk: int = 4):
        super().__init__()
        self.n1=n1; self.n2=n2; self.key_dim=key_dim; self.value_dim=value_dim; self.topk=topk
        self.k1 = nn.Parameter(torch.randn(n1,key_dim)*0.02)
        self.k2 = nn.Parameter(torch.randn(n2,key_dim)*0.02)
        self.values = nn.Parameter(torch.randn(n1*n2,value_dim)*0.01)
        self.q_proj = nn.Linear(d, 2*key_dim, bias=False)
        self.val_proj = nn.Linear(value_dim, d, bias=False)
        self.gate = nn.Linear(d,1)
        # discourage overuse at init
        with torch.no_grad():
            if self.gate.bias is None:
                self.gate.bias = nn.Parameter(torch.tensor([-2.0]))
            else:
                self.gate.bias.fill_(-2.0)
        self.dropout = nn.Dropout(p=0.05)
    def forward(self, x):  # x (B,T,d)
        B, T, D = x.shape
        q = self.q_proj(x)  # (B,T,2k)
        q1, q2 = q.split(self.key_dim, dim=-1)
        # scores
        s1 = torch.einsum('btk,nk->btn', q1, self.k1)  # (B,T,n1)
        s2 = torch.einsum('btk,nk->btn', q2, self.k2)  # (B,T,n2)
        top1 = s1.topk(self.topk, dim=-1)
        top2 = s2.topk(self.topk, dim=-1)
        cand_vals = []
        cand_scores = []
        for i in range(self.topk):
            for j in range(self.topk):
                idx1 = top1.indices[:, :, i]
                idx2 = top2.indices[:, :, j]
                combined_index = idx1 * self.n2 + idx2
                score = top1.values[:, :, i] + top2.values[:, :, j]
                cand_scores.append(score)
                val = self.values[combined_index]
                cand_vals.append(val)
        scores = torch.stack(cand_scores, dim=-1)  # (B,T,K^2)
        vals = torch.stack(cand_vals, dim=-2)      # (B,T,K^2,val_dim)
        w = torch.softmax(scores, dim=-1)
        retrieved = (w.unsqueeze(-1) * vals).sum(-2)  # (B,T,val_dim)
        out = self.val_proj(retrieved)
        g = torch.sigmoid(self.gate(x))
        # Expose mean gate for optional regularization
        self.last_gate_mean = g.mean().detach()
        return x + g * self.dropout(out)
