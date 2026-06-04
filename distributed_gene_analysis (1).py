# distributed_gene_analysis.py
# Task 1b – Distributed ML on the REAL Golub Leukemia Gene Expression Dataset
#
# ── DATASET SETUP ─────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/crawford/gene-expression
# You will get 3 files — place all three in the same folder as this script:
#
#   data_set_ALL_AML_train.csv       ← training gene expression (7129 genes × 38 patients)
#   data_set_ALL_AML_independent.csv ← test gene expression     (7129 genes × 34 patients)
#   actual.csv                       ← labels for all 72 patients (patient, cancer)
#
# NOTE: The Kaggle files are TRANSPOSED — genes are rows, patients are columns.
#       This script handles that automatically.
#
# Run:  mpirun -np 6 --hostfile hostfile python3 distributed_gene_analysis.py

from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ── MPI init ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Rank 0 loads and prepares the real Kaggle dataset ─────────────────────────
if rank == 0:
    print("[Rank 0] Loading real Golub leukemia dataset ...")

    # ── Load gene expression files (genes as rows, patients as columns) ────────
    df_train = pd.read_csv("data_set_ALL_AML_train.csv")
    df_test  = pd.read_csv("data_set_ALL_AML_independent.csv")

    # The Kaggle files have extra columns: 'Gene Description', 'Gene Accession Number'
    # and alternating 'call' columns (A/M/P flags). We only want numeric expression columns.
    def extract_expression(df):
        # Keep only columns that are purely numeric patient columns (not 'call' columns)
        expr_cols = [c for c in df.columns
                     if c not in ("Gene Description", "Gene Accession Number")
                     and "call" not in str(c).lower()]
        return df[expr_cols].T   # Transpose: rows=patients, cols=genes

    train_expr = extract_expression(df_train)   # shape: (38, 7129)
    test_expr  = extract_expression(df_test)    # shape: (34, 7129)

    # Combine train + test into full dataset (72 patients total)
    X = pd.concat([train_expr, test_expr], axis=0).values.astype(float)

    # ── Load labels ─────────────────────────────────────────────────────────────
    labels_df = pd.read_csv("actual.csv")
    # actual.csv columns: 'patient' (1-72), 'cancer' ('ALL' or 'AML')
    labels_df = labels_df.sort_values("patient").reset_index(drop=True)
    y = labels_df["cancer"].map({"ALL": 0, "AML": 1}).values   # ALL=0, AML=1

    print(f"[Rank 0] Dataset: {X.shape[0]} patients × {X.shape[1]} genes")
    print(f"[Rank 0] Labels — ALL: {(y==0).sum()}, AML: {(y==1).sum()}")
else:
    X = None
    y = None

# ── Broadcast to all ranks ────────────────────────────────────────────────────
X = comm.bcast(X, root=0)
y = comm.bcast(y, root=0)

# ── Each rank takes its own patient slice ─────────────────────────────────────
chunk = len(X) // size
start = rank * chunk
end   = start + chunk if rank != size - 1 else len(X)
X_local = X[start:end]
y_local = y[start:end]

print(f"[Rank {rank} | {MPI.Get_processor_name()}]  "
      f"Training on {len(X_local)} patients, {X.shape[1]} genes ...")

# ── Train Random Forest on local shard ────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_local, y_local)
local_score = clf.score(X_local, y_local)
print(f"[Rank {rank}]  Local score: {local_score:.4f}")

# ── Gather all scores at rank 0 ───────────────────────────────────────────────
scores = comm.gather(local_score, root=0)

if rank == 0:
    print("\n" + "="*50)
    print("Bioinformatics scores from all nodes:", [f"{s:.4f}" for s in scores])
    print(f"Average score:                        {np.mean(scores):.4f}")
    print("="*50)
