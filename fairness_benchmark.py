import math, time, csv, json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters

# ------------------ Param parity helper ------------------

def build_hydra_parity(target_params_m: float, base_cfg: HydraConfig) -> HydraConfig:
    """Search simple grid over d and moe_experts to approach target params (in millions)."""
    best = None
    d_grid = [256, 272, 288, 304, 320]
    experts_grid = [4, 6, 8]
    hidden = base_cfg.moe_hidden
    for d in d_grid:
        for ne in experts_grid:
            cfg = HydraConfig(d=d, vocab_size=base_cfg.vocab_size, n_blocks=base_cfg.n_blocks,
                              attn_every=base_cfg.attn_every, moe_experts=ne, moe_hidden=hidden,
                              chunk_size=base_cfg.chunk_size, fast_ssm=True, disable_moe=False,
                              disable_attn=base_cfg.disable_attn, vector_moe=True)
            m = ToyHydra(cfg)
            p = count_parameters(m)/1e6
            diff = abs(p - target_params_m)
            if best is None or diff < best[0]:
                best = (diff, cfg, p)
    return best[1]

# ------------------ FLOPs estimation (rough) ------------------

def estimate_transformer_flops_per_token(d: int, heads: int, seq_len: int, ffn_mult: int = 4) -> int:
    # Very rough: QKV (3 d^2) + proj (d^2) + attention scores (seq_len * d) + attn value mix (seq_len * d) + FFN (2 d * (ffn_mult d) + (ffn_mult d) * d)
    dense = 4 * d * d + 2 * ffn_mult * d * d
    attn = 2 * seq_len * d
    return dense + attn

def estimate_fastssm_flops_per_token(d: int, kernel: int = 8) -> int:
    # in_proj 2d^2 + depthwise conv (kernel per channel) + pointwise negligible + gating elementwise
    return 2 * d * d + d * kernel

def estimate_moe_flops_per_token(d: int, hidden: int, active_experts: int = 2) -> int:
    # SwiGLU: w_in d*hidden + w_gate d*hidden + elementwise + w_out hidden*d (count matmul mult-add ~2 flops per multiply not modeled precisely)
    return active_experts * (2 * d * hidden + hidden * d)

# ------------------ Benchmark ------------------
@torch.no_grad()
def throughput(model, vocab, device, seq_lens: List[int], B: int = 2, steps: int = 6, warmup: int = 2):
    model.eval()
    out = {}
    for L in seq_lens:
        x = torch.randint(0, vocab, (B, L), device=device)
        for _ in range(warmup):
            _ = model(x)
        if device.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(steps):
            _ = model(x)
        if device.startswith('cuda'): torch.cuda.synchronize()
        t1 = time.time()
        toks = B * L * steps
        peak_mem = torch.cuda.max_memory_allocated()/1e6 if device.startswith('cuda') else 0.0
        out[L] = {'toks_per_s': toks / (t1 - t0), 'ms_per_tok': 1000*(t1 - t0)/toks, 'peak_mem_MB': peak_mem}
    return out

# Training loop

def train_short(model, vocab, device, steps=2000, B=8, T=512, lr=3e-4, warmup=40, log_expert=False, expert_log_path=None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    expert_logs = []
    for step in range(steps):
        x = torch.randint(0, vocab, (B, T), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        # Aux load-balance if enabled
        if log_expert and hasattr(model, 'moe_stats') and model.moe_stats:
            lb_terms = []
            for stat in model.moe_stats:
                if 'usage' in stat and stat['usage'] is not None:
                    u = stat['usage']  # (E,)
                    # Target uniform 1/E, penalty: E * sum(u^2) (minimum at uniform) - 1
                    E = u.numel()
                    lb = (E * (u ** 2).sum() - 1.0)
                    lb_terms.append(lb)
            if lb_terms:
                lb_loss = torch.stack(lb_terms).mean()
                w = getattr(model.cfg, 'aux_load_balance_weight', 0.0)
                loss = loss + w * lb_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step < warmup:
            for g in opt.param_groups: g['lr'] = lr * (step + 1)/warmup
        else:
            progress = (step - warmup)/(steps - warmup)
            for g in opt.param_groups: g['lr'] = 0.1*lr + 0.9*lr*0.5*(1+math.cos(math.pi*progress))
        if (step+1) % 50 == 0:
            losses.append((step+1, float(loss.detach())))
        if log_expert and hasattr(model, 'moe_stats') and model.moe_stats:
            for stat in model.moe_stats:
                probs = stat['probs']
                p = probs.mean(dim=1)
                ent = -(p * (p+1e-9).log()).sum(-1).mean().item()
                usage = stat.get('usage', None)
                if usage is not None:
                    usage_std = float(usage.std().item())
                else:
                    usage_std = None
                expert_logs.append({'step': step+1, 'entropy': ent, 'usage_std': usage_std})
    if log_expert and expert_log_path:
        with open(expert_log_path, 'w') as f: json.dump(expert_logs, f, indent=2)
    return losses

# Variant builder

def build_variants(base_cfg: HydraConfig, transformer_layers: int):
    variants = {}
    # Baseline transformer
    variants['transformer_base'] = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=transformer_layers)
    # Hydra base
    variants['hydra_base'] = ToyHydra(base_cfg)
    # Hydra no MoE
    cfg_no_moe = HydraConfig(**{**base_cfg.__dict__, 'disable_moe': True})
    variants['hydra_no_moe'] = ToyHydra(cfg_no_moe)
    # Hydra no attention
    cfg_no_attn = HydraConfig(**{**base_cfg.__dict__, 'disable_attn': True})
    variants['hydra_no_attn'] = ToyHydra(cfg_no_attn)
    # Param parity
    target = count_parameters(variants['transformer_base'])/1e6
    parity_cfg = build_hydra_parity(target, base_cfg)
    variants['hydra_parity'] = ToyHydra(parity_cfg)
    return variants, parity_cfg

# Main
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_cfg = HydraConfig(d=256, n_blocks=8, attn_every=4, moe_experts=4, moe_hidden=256, vocab_size=4000)
    transformer_layers = 8
    seq_lens = [1024, 2048, 4096, 8192, 16384]

    variants, parity_cfg = build_variants(base_cfg, transformer_layers)

    # Report param counts
    print('Variant parameter counts (M):')
    for name, model in variants.items():
        print(f'  {name:18s}: {count_parameters(model)/1e6:6.2f}')
    print('\nParity config chosen:', parity_cfg)

    # Throughput
    print('\nThroughput tokens/s:')
    for name, model in variants.items():
        th = throughput(model.to(device), base_cfg.vocab_size, device, seq_lens)
        # Extract nested numeric field 'toks_per_s' for compact summary
        summary = {L: f"{v['toks_per_s']:.0f}" for L, v in th.items()}
        print(f'{name}: {summary}')
        # Optional detailed per-length line (uncomment if desired)
        # for L, v in th.items():
        #     print(f"  L={L}: {v['toks_per_s']:.0f} tok/s  {v['ms_per_tok']:.3f} ms/tok  peakMB={v['peak_mem_MB']:.1f}")
        model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None

    # Throughput CSV
    with open('results/throughput_summary.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','seq_len','toks_per_s','ms_per_tok','peak_mem_MB'])
        for name, model in variants.items():
            for L, v in throughput(model.to(device), base_cfg.vocab_size, device, seq_lens).items():
                writer.writerow([name, L, v['toks_per_s'], v['ms_per_tok'], v['peak_mem_MB']])
                model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None
    print('Saved throughput_summary.csv')

    # Train subset (skip full due to time) on base + parity + transformer
    train_subset = ['transformer_base', 'hydra_base', 'hydra_parity']
    with open('results/train_losses.csv','w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['model','step','loss'])
        for name in train_subset:
            model = variants[name].to(device)
            losses = train_short(model, base_cfg.vocab_size, device, log_expert=(name=='hydra_parity'), expert_log_path='results/expert_logs.json')
            for step_, loss_ in losses:
                writer.writerow([name, step_, loss_])
            model.to('cpu'); torch.cuda.empty_cache() if device.startswith('cuda') else None
    print('Saved train_losses.csv and expert_logs.json (for hydra_parity)')

    # Rough FLOPs per token comparison at largest seq length
    L = max(seq_lens)
    trans_flops = estimate_transformer_flops_per_token(base_cfg.d, base_cfg.n_heads, L)
    hydra_flops = estimate_fastssm_flops_per_token(base_cfg.d) + estimate_moe_flops_per_token(base_cfg.d, base_cfg.moe_hidden)
    print(f"\nApprox per-token FLOPs @L={L}: transformer_layer ~{trans_flops/1e6:.2f}M vs hydra_block ~{hydra_flops/1e6:.2f}M (ignoring attention layers).")
    print('Done.')
