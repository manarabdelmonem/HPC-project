# distributed_gene_expression_analysis.py
# Task 2 – Distributed ML with PySpark on the REAL Golub Leukemia Dataset
#
# ── DATASET SETUP ─────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/crawford/gene-expression
# Copy the 3 files into the spark-master container:
#
#   docker cp data_set_ALL_AML_train.csv       <spark-master-id>:/opt/spark/data/
#   docker cp data_set_ALL_AML_independent.csv <spark-master-id>:/opt/spark/data/
#   docker cp actual.csv                       <spark-master-id>:/opt/spark/data/
#
# Submit:
#   docker exec -it <spark-master-id> spark-submit \
#     --master spark://master:7077 \
#     /opt/spark/apps/distributed_gene_expression_analysis.py

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np

# ── Spark session ─────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Leukemia_Golub_RealDataset") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"Spark {spark.version} running on {spark.sparkContext.master}")

DATA_DIR = "/opt/spark/data"

# ── Load and prepare dataset using pandas, then convert to Spark ──────────────
# (Kaggle files are transposed: genes=rows, patients=columns — easier to fix in pandas)
print("Loading real Golub dataset ...")

def extract_expression(path):
    df = pd.read_csv(path)
    expr_cols = [c for c in df.columns
                 if c not in ("Gene Description", "Gene Accession Number")
                 and "call" not in str(c).lower()]
    return df[expr_cols].T.reset_index(drop=True)

train_expr = extract_expression(f"{DATA_DIR}/data_set_ALL_AML_train.csv")
test_expr  = extract_expression(f"{DATA_DIR}/data_set_ALL_AML_independent.csv")
X_all = pd.concat([train_expr, test_expr], axis=0).reset_index(drop=True)
X_all.columns = [f"gene_{i}" for i in range(X_all.shape[1])]
X_all = X_all.astype(float)

# Labels
labels_df = pd.read_csv(f"{DATA_DIR}/actual.csv")
labels_df = labels_df.sort_values("patient").reset_index(drop=True)
X_all["label"] = labels_df["cancer"].map({"ALL": 0.0, "AML": 1.0}).values

print(f"Dataset: {X_all.shape[0]} patients × {X_all.shape[1]-1} genes")
print(X_all["label"].value_counts().rename({0.0: "ALL", 1.0: "AML"}).to_string())

# ── Convert to Spark DataFrame ────────────────────────────────────────────────
df = spark.createDataFrame(X_all)

# ── Assemble gene columns into one feature vector ─────────────────────────────
gene_cols = [c for c in df.columns if c.startswith("gene_")]
assembler = VectorAssembler(inputCols=gene_cols, outputCol="features")
data      = assembler.transform(df).select("features", "label")

# ── 80/20 split ───────────────────────────────────────────────────────────────
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_data.count()} patients  |  Test: {test_data.count()} patients")

# ── Train RandomForest ────────────────────────────────────────────────────────
rf    = RandomForestClassifier(labelCol="label", featuresCol="features",
                               numTrees=100, seed=42)
model = rf.fit(train_data)

# ── Evaluate ──────────────────────────────────────────────────────────────────
predictions = model.transform(test_data)
correct  = predictions.filter(predictions["label"] == predictions["prediction"]).count()
accuracy = correct / test_data.count()

print("\n" + "="*50)
print(f"Accuracy on test data:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print("="*50)

spark.stop()
print("Done.")
