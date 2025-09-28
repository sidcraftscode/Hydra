"""Benchmark long context efficiency with and without workspace/PKM (enhanced).

Adds optional throughput capping so we can compare peak memory at a constant
tokens-per-second rate across models. Use --cap-tps to enforce a max toks/s.
"""
import torch, time, json, statistics as stats
import argparse
from toy_hydra import HydraConfig, ToyHydra, BaselineTransformer, count_parameters

@torch.no_grad()
def bench(model, vocab, lens, device='cuda', warmup=3, runs=5, cap_toks_per_s=None):
    model.eval()
    results = {}
    for L in lens:
        x = torch.randint(0, vocab, (1, L), device=device)
        # warmup runs
        for _ in range(warmup):
            _ = model(x)
        if device.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times_effective = []
        toks = L
        for _ in range(runs):
            t0 = time.time(); _ = model(x)
            if device.startswith('cuda'): torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            # Enforce pacing to cap throughput (sleep if faster than cap)
            if cap_toks_per_s is not None and cap_toks_per_s > 0:
                target_dt = toks / float(cap_toks_per_s)
                if dt < target_dt:
                    time.sleep(target_dt - dt)
                    dt = target_dt
            times_effective.append(dt)
        toks_per_s = [toks / dt for dt in times_effective]
        ms_per_tok = [1000*dt / toks for dt in times_effective]
        peak_mem = torch.cuda.max_memory_allocated()/1e6 if device.startswith('cuda') else 0.0
        results[L] = {
            'toks_s_mean': stats.mean(toks_per_s),
            'toks_s_std': stats.pstdev(toks_per_s),
            'ms_tok_mean': stats.mean(ms_per_tok),
            'ms_tok_std': stats.pstdev(ms_per_tok),
            'peak_mem_MB': peak_mem
        }
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Long-context benchmark with optional throughput cap')
    parser.add_argument('--cap-tps', type=float, default=3000, help='Cap throughput to this tokens/sec by pacing with sleep')
    parser.add_argument('--runs', type=int, default=5, help='Number of timed runs per length')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs per length')
    parser.add_argument('--device', type=str, default=None, help='cpu or cuda (auto if not set)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    lens = [1024,2048,4096,8192,16384]
    base_cfg = HydraConfig()
    mem_cfg = HydraConfig()
    if hasattr(base_cfg, 'use_workspace'):
        base_cfg.use_workspace = False; base_cfg.use_pkm = False
        mem_cfg.use_workspace = True; mem_cfg.use_pkm = True
    h_base = ToyHydra(base_cfg).to(device)
    h_mem = ToyHydra(mem_cfg).to(device)
    trans = BaselineTransformer(d=base_cfg.d, vocab_size=base_cfg.vocab_size, n_layers=base_cfg.n_blocks).to(device)
    print('Params Hydra base / mem / transformer:', f'{count_parameters(h_base)/1e6:.2f}', f'{count_parameters(h_mem)/1e6:.2f}', f'{count_parameters(trans)/1e6:.2f}')
    if args.cap_tps:
        print(f'Pacing to cap throughput at {args.cap_tps} tokens/sec (sleep-based).')
    base_res = bench(h_base, base_cfg.vocab_size, lens, device, warmup=args.warmup, runs=args.runs, cap_toks_per_s=args.cap_tps)
    mem_res = bench(h_mem, base_cfg.vocab_size, lens, device, warmup=args.warmup, runs=args.runs, cap_toks_per_s=args.cap_tps)
    trans_res = bench(trans, base_cfg.vocab_size, lens, device, warmup=args.warmup, runs=args.runs, cap_toks_per_s=args.cap_tps)
    def pretty(name,res):
        print(f'\n{name}:')
        for L,v in res.items():
            print(f'L={L:5d}  {v["toks_s_mean"]:10.1f}±{v["toks_s_std"]:6.1f} toks/s  {v["ms_tok_mean"]:6.3f}±{v["ms_tok_std"]:5.3f} ms/token  peakMB={v["peak_mem_MB"]:.1f}')
    pretty('Hydra base', base_res)
    pretty('Hydra + memory', mem_res)
    pretty('Transformer', trans_res)
    all_json = {'hydra_base': base_res, 'hydra_memory': mem_res, 'transformer': trans_res}
    with open('results/long_context_stats.json','w') as f:
        json.dump(all_json, f, indent=2)
    print('\nSaved JSON: results/long_context_stats.json')
