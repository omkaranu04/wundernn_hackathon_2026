"""Blend two soft-target files: out = (1-w) * A + w * B (float16 memmaps, same shape), in chunks.
    python blend_soft.py soft/train_TV20L.npy soft/train_LA25.npy 0.5 soft/train_TV20L_LA25.npy"""
import sys
import numpy as np

a, b, w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
A = np.load(a, mmap_mode="r"); B = np.load(b, mmap_mode="r")
assert A.shape == B.shape, (A.shape, B.shape)
O = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=A.shape)
for i in range(0, A.shape[0], 256):
    O[i:i + 256] = ((1 - w) * A[i:i + 256].astype(np.float32) + w * B[i:i + 256].astype(np.float32)).astype(np.float16)
O.flush()
print(f"wrote {out}: {A.shape}, weight {w} on {b}")
