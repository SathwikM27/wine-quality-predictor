import os
import subprocess
try:
    import numpy
except ImportError:
    print("NumPy is not installed. Installing now...")
    subprocess.check_call(["pip", "install", "--user", "numpy"])

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .master("local[*]") \
    .getOrCreate()

# Paths
test_path = "hdfs://ip-172-31-74-189.ec2.internal:8020/data/validation.csv"  # Ensure the Docker container or local script uses this path
model_path = "hdfs://ip-172-31-74-189.ec2.internal:8020/models/wine_quality_model"

# Load test data
test_data = spark.read.csv(test_path, header=True, inferSchema=True, sep=";", quote='"')
test_data = test_data.toDF(*[col.strip('"').strip() for col in test_data.columns])

# Assemble features
feature_columns = [col for col in test_data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_data = assembler.transform(test_data)

# Load trained Logistic Regression model
model = LogisticRegressionModel.load(model_path)

# Predict and Evaluate
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

spark.stop()
