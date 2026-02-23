"""
Convert PyTorch .pth checkpoint to Jittor-compatible .pkl format.
Only requires torch + numpy (no jittor needed), so run in open-mmlab env:
  python.exe tools/convert_pth_to_pkl.py <input.pth> [output.pkl]
"""
import sys
import os
import pickle
import numpy as np
import torch


def convert(pth_path, pkl_path):
    print(f"[INFO] Loading: {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu')

    # Handle mmdet-style checkpoint formats
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        meta = ckpt.get('meta', {})
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        meta = ckpt.get('meta', {})
    else:
        state_dict = ckpt
        meta = {}

    # Convert torch.Tensor → numpy (Jittor can load these with jt.array)
    np_state_dict = {}
    for k, v in state_dict.items():
        np_state_dict[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

    # Write pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump({'state_dict': np_state_dict, 'meta': meta}, f)

    size_mb = os.path.getsize(pkl_path) / 1024 / 1024
    print(f"[INFO] Saved: {pkl_path}  ({size_mb:.1f} MB,  {len(np_state_dict)} params)")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        inp = sys.argv[1]
        out = os.path.splitext(inp)[0] + '.pkl'
        convert(inp, out)
    else:
        print("Usage: python convert_pth_to_pkl.py <input.pth> [output.pkl]")
