# distributed_gene_expression_analysis.py
# Task 2 – Distributed Bioinformatics ML with Apache Spark MLlib
#
# Deploy the Spark cluster first:
#   docker stack deploy -c spark-stack.yml spark
#
# Submit this job:
#   docker exec -it <spark-master-container-id> \
#     spark-submit --master spark://master:7077 \
#     distributed_gene_expression_analysis.py
#
# Monitor: http://192.168.56.10:8080

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# ── Initialize Spark Session ─────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("BioinformaticsML") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")
print(f"Master: {spark.sparkContext.master}")

# ── Load leukemia gene expression CSV ────────────────────────────────────────
# File must be accessible inside the container at this path
df = spark.read.csv("leukemia_expression.csv", header=True, inferSchema=True)

print(f"Dataset: {df.count()} rows × {len(df.columns)} columns")
df.groupBy(df.columns[-1]).count().show()

# ── Assemble all gene columns into a single feature vector ───────────────────
# All columns except the last are gene expression features
feature_cols = df.columns[:-1]
assembler    = VectorAssembler(inputCols=feature_cols, outputCol="features")
data         = assembler.transform(df)

# ── Split into 80% train / 20% test ─────────────────────────────────────────
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_data.count()} rows  |  Test: {test_data.count()} rows")

# ── Train a Random Forest classifier ─────────────────────────────────────────
# labelCol = the last column (AML=0, ALL=1)
rf = RandomForestClassifier(
    labelCol=df.columns[-1],
    featuresCol="features",
    numTrees=100,
    seed=42
)

model = rf.fit(train_data)

# ── Evaluate on test set ─────────────────────────────────────────────────────
predictions = model.transform(test_data)

# Count rows where true label matches prediction
correct  = predictions.filter(
    predictions[df.columns[-1]] == predictions["prediction"]
).count()
accuracy = correct / test_data.count()

print("\n" + "="*50)
print(f"Accuracy on test data: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print("="*50)

# ── Stop Spark session ───────────────────────────────────────────────────────
spark.stop()
print("Spark session stopped.")
