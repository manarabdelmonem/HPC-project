# distributed_mnist.py
# Task 1 – Distributed ML on the Digits Dataset using MPI (mpi4py)
#
# Cluster specs:
#   master   192.168.56.10  5 GB RAM  5 cores  Rank 0
#   worker1  192.168.56.11  4 GB RAM  4 cores  Rank 1
#   worker2  192.168.56.12  4 GB RAM  3 cores  Rank 2
#
# Hostfile (slots match tutorial: 2 per node):
#   master slots=2  |  worker1 slots=2  |  worker2 slots=2
#
# How to run (from master):
#   source ~/venvs/hpc_ml/bin/activate
#   mpirun -np 6 --hostfile hostfile python3 distributed_mnist.py

from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ── Initialize MPI communication ─────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # this process's rank (0 = master, 1–5 = workers)
size = comm.Get_size()   # total number of MPI processes

# ── Rank 0 loads the dataset; others wait ────────────────────────────────────
if rank == 0:
    digits = load_digits()          # 1,797 samples, 64 features, 10 classes
    data   = digits.data            # shape: (1797, 64)
    target = digits.target          # shape: (1797,)
    print(f"[Rank 0] Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
else:
    data   = None
    target = None

# ── Broadcast full dataset to all ranks ──────────────────────────────────────
# Every rank receives a copy so each can slice its own chunk
data   = comm.bcast(data,   root=0)
target = comm.bcast(target, root=0)

# ── Each rank computes its own contiguous chunk ───────────────────────────────
chunk_size  = len(data) // size
start       = rank * chunk_size
# Last rank takes any leftover samples
end         = start + chunk_size if rank != size - 1 else len(data)
local_data   = data[start:end]
local_target = target[start:end]

print(f"[Rank {rank} | {MPI.Get_processor_name()}]  "
      f"Training on samples {start}–{end-1}  ({len(local_data)} samples)")

# ── Each rank trains a local Random Forest on its chunk ───────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(local_data, local_target)

# ── Score locally on the same chunk ──────────────────────────────────────────
local_score = clf.score(local_data, local_target)
print(f"[Rank {rank}]  Local score: {local_score:.4f}")

# ── Gather all local scores at rank 0 ────────────────────────────────────────
scores = comm.gather(local_score, root=0)

# ── Rank 0 prints the aggregated results ─────────────────────────────────────
if rank == 0:
    print("\n" + "="*50)
    print("Scores from all nodes:", [f"{s:.4f}" for s in scores])
    print(f"Average score:          {np.mean(scores):.4f}")
    print("="*50)
    print("\nCluster used:")
    print("  master  (192.168.56.10) — 5 GB RAM, 5 cores")
    print("  worker1 (192.168.56.11) — 4 GB RAM, 4 cores")
    print("  worker2 (192.168.56.12) — 4 GB RAM, 3 cores")
