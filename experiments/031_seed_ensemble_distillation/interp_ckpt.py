"""043: weight-space interpolation (WiSE-FT): OUT/ema.pt = base + A * (FT/last.pt - base). Usage: interp_ckpt.py BASE FT A OUT"""
import sys, json, shutil, pathlib, torch
B, F, A, O = sys.argv[1], sys.argv[2], float(sys.argv[3]), pathlib.Path('results/checkpoints_' + sys.argv[4])
b = torch.load(f'results/checkpoints_{B}/ema.pt', map_location='cpu'); f = torch.load(f'results/checkpoints_{F}/last.pt', map_location='cpu')
O.mkdir(parents=True, exist_ok=True); shutil.copy(f'results/checkpoints_{F}/config.json', O / 'config.json')
torch.save({k: b[k] + A * (f[k] - b[k]) if b[k].is_floating_point() else f[k] for k in b}, O / 'ema.pt')
print(f'{O}: {B} + {A} x ({F}/last - base)')
