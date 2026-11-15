"""038 (2026-11-06, adversarial-ideation idea 3): per-tick-unit look-ahead window in the LSTM's labels.
Sequences of the 0.6569 unit take the LA25 label (soft/*_TV12_LA25.npy), every other unit keeps LA12 (soft/*_TV12_LA12.npy).
Per-unit honest teacher WP (fold 0) on 0.6569: LA12 0.545 -> LA25 0.602; the GRU partner keeps LA50, so the pair's
labels stay different everywhere.
    python unit_window_soft.py   ->  soft/{train,valid}_TV12_LAU.npy
"""
import numpy as np

for split in ("train", "valid"):
    A = np.load(f"soft/{split}_TV12_LA12.npy", mmap_mode="r"); B = np.load(f"soft/{split}_TV12_LA25.npy", mmap_mode="r")
    U = np.load(f"soft/units_{split}.npy")
    assert A.shape == B.shape and A.shape[0] == len(U)
    O = np.lib.format.open_memmap(f"soft/{split}_TV12_LAU.npy", mode="w+", dtype=np.float16, shape=A.shape)
    for i in range(A.shape[0]):
        O[i] = B[i] if U[i] == 3 else A[i]
    O.flush()
    print(f"wrote soft/{split}_TV12_LAU.npy: {A.shape}, LA25 on {(U == 3).sum()} of {len(U)} sequences")
