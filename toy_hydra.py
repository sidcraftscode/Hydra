import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from workspace_memory import WorkspaceMemory
from pkm_memory import PKMMemory

# ------------------ Utility ------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------ SSM Path (simplified selective state) ------------------
class SimpleSSM(nn.Module):
    """Original slow recurrent SSM (kept for reference / correctness)."""
    def __init__(self, d: int, use_skip: bool = True):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(d, 4 * d, bias=False)
        self.lambda_base = nn.Parameter(torch.ones(d))
        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Linear(d, d, bias=False)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, x):
        B, T, D = x.shape
        params = self.proj(x)
        delta, B_t, C_t, U_t = params.chunk(4, dim=-1)
        alpha_t = torch.exp(-F.softplus(delta) * F.softplus(self.lambda_base))
        s = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):  # Python loop = slow on CPU
            y_t = C_t[:, t] * s
            if self.use_skip:
                y_t = y_t + self.skip(x[:, t])
            outs.append(y_t)
            s = alpha_t[:, t] * s + B_t[:, t] * U_t[:, t]
        y = torch.stack(outs, dim=1)
        return self.out_ln(y)

class FastSSM(nn.Module):
    """Fast linear-time surrogate for SSM using gated depthwise conv (approximates selective scan behavior).
    This is used in benchmarking to remove Python loop overhead so we can showcase scaling.
    Not mathematically identical but preserves linear complexity + gating idea."""
    def __init__(self, d: int, kernel_size: int = 8):
        super().__init__()
        self.d = d
        import math
        import time
        from dataclasses import dataclass
        from typing import Optional, List, Tuple

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from workspace_memory import WorkspaceMemory
        from pkm_memory import PKMMemory

        # ------------------ Utility ------------------

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ------------------ SSM Path (simplified selective state) ------------------
        class SimpleSSM(nn.Module):
            """Original slow recurrent SSM (kept for reference / correctness)."""
            def __init__(self, d: int, use_skip: bool = True):
                super().__init__()
                self.d = d
                self.proj = nn.Linear(d, 4 * d, bias=False)
                self.lambda_base = nn.Parameter(torch.ones(d))
                self.use_skip = use_skip
                if use_skip:
                    self.skip = nn.Linear(d, d, bias=False)
                self.out_ln = nn.LayerNorm(d)

            def forward(self, x):
                B, T, D = x.shape
                params = self.proj(x)
                delta, B_t, C_t, U_t = params.chunk(4, dim=-1)
                alpha_t = torch.exp(-F.softplus(delta) * F.softplus(self.lambda_base))
                s = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
                outs = []
                for t in range(T):  # Python loop = slow on CPU
                    y_t = C_t[:, t] * s
                    if self.use_skip:
                        y_t = y_t + self.skip(x[:, t])
                    outs.append(y_t)
                    s = alpha_t[:, t] * s + B_t[:, t] * U_t[:, t]
                y = torch.stack(outs, dim=1)
                return self.out_ln(y)

        class FastSSM(nn.Module):
            """Fast linear-time surrogate for SSM using gated depthwise conv (approximates selective scan behavior).
            This is used in benchmarking to remove Python loop overhead so we can showcase scaling.
            Not mathematically identical but preserves linear complexity + gating idea."""
            def __init__(self, d: int, kernel_size: int = 8):
                super().__init__()
                self.d = d
                self.kernel_size = kernel_size
                self.in_proj = nn.Linear(d, 2 * d, bias=False)  # value + gate
                # depthwise conv over value stream
                self.dw_conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size - 1, groups=d, bias=False)
                self.out_ln = nn.LayerNorm(d)
                nn.init.zeros_(self.dw_conv.weight)  # start near identity

            def forward(self, x):  # (B,T,d)
                v, g = self.in_proj(x).chunk(2, dim=-1)
                g = torch.sigmoid(g)
                # depthwise conv expects (B,d,T)
                v_conv = self.dw_conv(v.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
                y = g * v_conv + (1 - g) * v  # gated blend
                return self.out_ln(y)

        # ------------------ Attention (sparse global) ------------------
        class SparseGlobalAttention(nn.Module):
            def __init__(self, d: int, num_heads: int = 4, window: int = 256, global_topk: int = 0):
                super().__init__()
                self.d = d
                self.h = num_heads
                self.head_dim = d // num_heads
                assert d % num_heads == 0
                self.qkv = nn.Linear(d, 3 * d, bias=False)
                self.o = nn.Linear(d, d, bias=False)
                self.window = window
                self.global_topk = global_topk
                self.ln = nn.LayerNorm(d)

            def forward(self, x):
                B, T, D = x.shape
                x = self.ln(x)
                qkv = self.qkv(x)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, T, self.h, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.h, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.h, self.head_dim).transpose(1, 2)
                attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                if self.window is not None:
                    idx = torch.arange(T, device=x.device)
                    local = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= self.window
                    mask = mask & local
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
                attn = F.softmax(attn_scores, dim=-1)
                out = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2).contiguous().view(B, T, D)
                return self.o(out)

        # ------------------ MoE ------------------
        class SwiGLU(nn.Module):
            def __init__(self, d_in: int, d_hidden: int):
                super().__init__()
                self.w_in = nn.Linear(d_in, d_hidden, bias=False)
                self.w_gate = nn.Linear(d_in, d_hidden, bias=False)
                self.w_out = nn.Linear(d_hidden, d_in, bias=False)
            def forward(self, x):
                return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))

        class ChunkTop2MoE(nn.Module):
            def __init__(self, d: int, n_experts: int = 4, hidden: int = 192, chunk_size: int = 32, top_k: int = 2, full_parallel: bool = True, temp: float = 1.0):
                super().__init__()
                self.d = d
                self.n_experts = n_experts
                self.chunk_size = chunk_size
                self.top_k = min(top_k, n_experts)
                self.full_parallel = full_parallel  # if True compute all experts then weight (fast for small n_experts)
                self.experts = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_experts)])
                self.gate = nn.Linear(d, n_experts, bias=False)
                self.ln = nn.LayerNorm(d)
                self.temp = temp
                # stats buffers
                self.last_logits = None
                self.last_top_indices = None
                self.last_probs = None
                self.last_usage = None  # per-expert usage distribution
            def forward(self, x):  # (B,T,d)
                B, T, D = x.shape
                x_norm = self.ln(x)
                if T % self.chunk_size != 0:
                    pad = self.chunk_size - (T % self.chunk_size)
                    x_norm = F.pad(x_norm, (0,0,0,pad))
                Tp = x_norm.size(1)
                C = Tp // self.chunk_size
                x_chunks = x_norm.view(B, C, self.chunk_size, D)
                chunk_means = x_chunks.mean(dim=2)  # (B,C,d)
                logits = self.gate(chunk_means) / self.temp  # temperature scaling
                topv, topi = logits.topk(self.top_k, dim=-1)  # (B,C,top_k)
                probs = F.softmax(topv, dim=-1)               # (B,C,top_k)
                self.last_logits = logits.detach()
                self.last_top_indices = topi.detach()
                self.last_probs = probs.detach()
                if self.full_parallel:
                    expert_outputs = []
                    flat = x_norm  # (B,Tp,d)
                    for e in self.experts:
                        expert_outputs.append(e(flat))
                    expert_outputs = torch.stack(expert_outputs, dim=0)  # (E,B,Tp,d)
                    weights = torch.zeros(B, C, self.n_experts, device=x.device, dtype=x.dtype)
                    weights.scatter_add_(2, topi, probs)  # place probs into selected experts
                    # usage distribution (sum over batch+chunks)
                    usage = weights.sum(dim=(0,1)) + 1e-9  # (E,)
                    self.last_usage = (usage / usage.sum()).detach()
                    weights_tokens = weights.unsqueeze(3).expand(B, C, self.n_experts, self.chunk_size)
                    weights_tokens = weights_tokens.reshape(B, self.n_experts, Tp)  # (B,E,Tp)
                    out = (weights_tokens.unsqueeze(-1) * expert_outputs.permute(1,0,2,3)).sum(dim=1)  # (B,Tp,d)
                    out = out[:, :T]
                else:
                    out = torch.zeros(B, Tp, D, device=x.device, dtype=x.dtype)
                    usage_counts = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
                    for k_idx in range(self.top_k):
                        e_indices = topi[:, :, k_idx]
                        p = probs[:, :, k_idx]
                        for e in range(self.n_experts):
                            mask = (e_indices == e)
                            if not mask.any():
                                continue
                            usage_counts[e] += mask.sum()
                            token_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(B, C, self.chunk_size, D)
                            tokens = x_chunks[token_mask].view(-1, D)
                            y = self.experts[e](tokens)
                            out[token_mask] += (p.unsqueeze(-1).unsqueeze(-1).expand_as(x_chunks))[token_mask].view(-1,1) * y
                    self.last_usage = (usage_counts + 1e-9) / (usage_counts.sum() + 1e-9)
                    out = out[:, :T]
                return out

        # ------------------ Hydra Block ------------------
        class HydraBlock(nn.Module):
            def __init__(self, d: int, use_attention: bool, use_moe: bool, n_heads=4, moe_experts=4, moe_hidden=192, chunk_size=32, fast_ssm=True, vector_moe=True, gate_temp: float = 1.0):
                super().__init__()
                self.ssm = FastSSM(d) if fast_ssm else SimpleSSM(d)
                self.attn = SparseGlobalAttention(d, num_heads=n_heads, window=256) if use_attention else None
                self.moe = ChunkTop2MoE(d, n_experts=moe_experts, hidden=moe_hidden, chunk_size=chunk_size, full_parallel=vector_moe, temp=gate_temp) if use_moe else None
                self.g1 = nn.Parameter(torch.tensor(1.0))
                self.g2 = nn.Parameter(torch.tensor(0.15 if use_attention else 0.0))
                self.g3 = nn.Parameter(torch.tensor(0.5 if use_moe else 0.0))
                self.res_ln = nn.LayerNorm(d)
            def forward(self, x):
                ssm_out = self.ssm(x) * self.g1
                if self.attn is not None:
                    attn_out = self.attn(x) * self.g2
                else:
                    attn_out = 0.0
                if self.moe is not None:
                    moe_out = self.moe(x) * self.g3
                else:
                    moe_out = 0.0
                x = x + ssm_out + attn_out + moe_out
                return self.res_ln(x)

        # ------------------ Hydra Model ------------------
        @dataclass
        class HydraConfig:
            d: int = 256
            vocab_size: int = 4000
            n_blocks: int = 8
            attn_every: int = 4
            moe_on: List[int] = None
            n_heads: int = 4
            moe_experts: int = 4
            moe_hidden: int = 256  # slightly larger hidden for quality
            chunk_size: int = 32
            fast_ssm: bool = True
            disable_moe: bool = False
            disable_attn: bool = False
            vector_moe: bool = True  # new flag
            use_workspace: bool = False
            use_pkm: bool = False
            gate_temp: float = 1.0  # new: temperature for MoE gate logits
            aux_load_balance_weight: float = 0.01  # coefficient for load balance loss

        class ToyHydra(nn.Module):
            def __init__(self, cfg: HydraConfig):
                super().__init__()
                self.cfg = cfg
                self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
                if cfg.moe_on is None:
                    moe_on = set(range(0, cfg.n_blocks, 2))
                else:
                    moe_on = set(cfg.moe_on)
                blocks = []
                for i in range(cfg.n_blocks):
                    use_attn = (not cfg.disable_attn) and ((i + 1) % cfg.attn_every == 0)
                    use_moe = (not cfg.disable_moe) and (i in moe_on)
                    blocks.append(HydraBlock(cfg.d, use_attn, use_moe, n_heads=cfg.n_heads,
                                             moe_experts=cfg.moe_experts, moe_hidden=cfg.moe_hidden,
                                             chunk_size=cfg.chunk_size, fast_ssm=cfg.fast_ssm, vector_moe=cfg.vector_moe,
                                             gate_temp=cfg.gate_temp))
                self.blocks = nn.ModuleList(blocks)
                self.workspace = WorkspaceMemory(cfg.d) if cfg.use_workspace else None
                self.pkm = PKMMemory(cfg.d) if cfg.use_pkm else None
                self.ln_f = nn.LayerNorm(cfg.d)
                self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
                self.head.weight = self.embed.weight
            def forward(self, idx):
                x = self.embed(idx)
                self.moe_stats = []
                for blk in self.blocks:
                    x = blk(x)
                    if getattr(blk, 'moe', None) is not None and blk.moe.last_logits is not None:
                        self.moe_stats.append({
                            'logits': blk.moe.last_logits,
                            'top_idx': blk.moe.last_top_indices,
                            'probs': blk.moe.last_probs,
                            'usage': blk.moe.last_usage
                        })
                    if self.workspace is not None:
                        x = self.workspace(x)
                    if self.pkm is not None:
                        x = self.pkm(x)
                x = self.ln_f(x)
                return self.head(x)

        # ------------------ Baseline Transformer ------------------
        class TransformerFFN(nn.Module):
            def __init__(self, d, mult=4):
                super().__init__()
                self.w1 = nn.Linear(d, mult * d, bias=False)
                self.w2 = nn.Linear(mult * d, d, bias=False)
            def forward(self, x):
                return self.w2(F.gelu(self.w1(x)))

        class CausalMHA(nn.Module):
            def __init__(self, d, n_heads=4):
                super().__init__()
                self.d = d
                self.h = n_heads
                self.head_dim = d // n_heads
                assert d % n_heads == 0
                self.qkv = nn.Linear(d, 3 * d, bias=False)
                self.o = nn.Linear(d, d, bias=False)
                self.ln = nn.LayerNorm(d)
            def forward(self, x):
                B, T, D = x.shape
                x_n = self.ln(x)
                qkv = self.qkv(x_n)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B, T, self.h, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.h, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.h, self.head_dim).transpose(1, 2)
                attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)
                mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                attn = attn.masked_fill(~mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                out = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2).contiguous().view(B, T, D)
                return x + self.o(out)

        class TransformerBlock(nn.Module):
            def __init__(self, d, n_heads=4, ffn_mult=4):
                super().__init__()
                self.mha = CausalMHA(d, n_heads)
                self.ffn_ln = nn.LayerNorm(d)
                self.ffn = TransformerFFN(d, mult=ffn_mult)
                self.res_ln = nn.LayerNorm(d)
            def forward(self, x):
                x = self.mha(x)
                x = x + self.ffn(self.ffn_ln(x))
                return self.res_ln(x)

        class BaselineTransformer(nn.Module):
            def __init__(self, d=256, vocab_size=4000, n_layers=8, n_heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d)
                self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(n_layers)])
                self.ln_f = nn.LayerNorm(d)
                self.head = nn.Linear(d, vocab_size, bias=False)
                self.head.weight = self.embed.weight
            def forward(self, idx):
                x = self.embed(idx)
                for b in self.blocks:
                    x = b(x)
                x = self.ln_f(x)
                return self.head(x)

        # ------------------ Quick self-test ------------------
        if __name__ == "__main__":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg = HydraConfig()
            hydra = ToyHydra(cfg).to(device)
            trans = BaselineTransformer(d=cfg.d, vocab_size=cfg.vocab_size, n_layers=cfg.n_blocks).to(device)
            print('Hydra params:', count_parameters(hydra)/1e6)
            print('Transformer params:', count_parameters(trans)/1e6)
            B, T = 2, 256
            x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
            y1 = hydra(x)
            y2 = trans(x)
            print('Output shapes', y1.shape, y2.shape)
