# distributed_gene_analysis.py
# Task 1 – Distributed Bioinformatics ML on Leukemia Gene Expression Data (MPI)
#
# Dataset: bioinfo_data/leukemia_expression.csv
#   200 samples × 500 gene features  |  Label: 0 = AML, 1 = ALL
#
# Cluster specs:
#   master   192.168.56.10  5 GB RAM  5 cores  Rank 0
#   worker1  192.168.56.11  4 GB RAM  4 cores  Rank 1
#   worker2  192.168.56.12  4 GB RAM  3 cores  Rank 2
#
# How to run (from master):
#   source ~/venvs/hpc_ml/bin/activate
#   mpirun -np 6 --hostfile hostfile python3 distributed_gene_analysis.py

from mpi4py import MPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ── Initialize MPI communication ─────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # this process's rank
size = comm.Get_size()   # total number of MPI processes

# ── Rank 0 loads the CSV; others wait ────────────────────────────────────────
if rank == 0:
    df     = pd.read_csv("bioinfo_data/leukemia_expression.csv")
    # All columns except the last are gene expression features
    data   = df.iloc[:, 1:-1].values.astype(float)   # drop sample_id col
    # Last column is the class label (0=AML, 1=ALL)
    target = df.iloc[:, -1].values.astype(int)
    print(f"[Rank 0] Dataset loaded: {data.shape[0]} samples, {data.shape[1]} genes")
    print(f"[Rank 0] Class counts — AML: {(target==0).sum()}, ALL: {(target==1).sum()}")
else:
    data   = None
    target = None

# ── Broadcast full dataset to all ranks ──────────────────────────────────────
data   = comm.bcast(data,   root=0)
target = comm.bcast(target, root=0)

# ── Each rank computes its own contiguous chunk of samples ───────────────────
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
    print("Bioinformatics scores from all nodes:", [f"{s:.4f}" for s in scores])
    print(f"Average score:                        {np.mean(scores):.4f}")
    print("="*50)
