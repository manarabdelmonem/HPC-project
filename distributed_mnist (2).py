# distributed_mnist.py
# Task 1a – Distributed ML on the Scikit-learn Digits Dataset using MPI
#
# ✅ No download needed — Digits is built into scikit-learn
#
# Cluster: master(192.168.56.10) worker1(192.168.56.11) worker2(192.168.56.12)
# Run:  mpirun -np 6 --hostfile hostfile python3 distributed_mnist.py

from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ── MPI init ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Rank 0 loads dataset ──────────────────────────────────────────────────────
if rank == 0:
    digits = load_digits()          # built-in: 1797 samples, 64 features, 10 classes
    data   = digits.data
    target = digits.target
    print(f"[Rank 0] Loaded Digits: {data.shape[0]} samples, {data.shape[1]} features")
else:
    data   = None
    target = None

# ── Broadcast full dataset to every rank ─────────────────────────────────────
data   = comm.bcast(data,   root=0)
target = comm.bcast(target, root=0)

# ── Each rank takes its own slice ─────────────────────────────────────────────
chunk = len(data) // size
start = rank * chunk
end   = start + chunk if rank != size - 1 else len(data)
local_data   = data[start:end]
local_target = target[start:end]

print(f"[Rank {rank} | {MPI.Get_processor_name()}]  Training on {len(local_data)} samples ...")

# ── Train local Random Forest ─────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(local_data, local_target)
local_score = clf.score(local_data, local_target)
print(f"[Rank {rank}]  Local score: {local_score:.4f}")

# ── Gather scores at rank 0 ───────────────────────────────────────────────────
scores = comm.gather(local_score, root=0)

if rank == 0:
    print("\n" + "="*50)
    print("Scores from all nodes:", [f"{s:.4f}" for s in scores])
    print(f"Average score:         {np.mean(scores):.4f}")
    print("="*50)
