import os
import subprocess
import sys

os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.local/bin")

# Set PYSPARK_PYTHON to the current Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable

try:
    import numpy
except ImportError:
    print("NumPy is not installed. Installing now...")
    subprocess.check_call(["pip", "install", "--user", "numpy"])

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TrainWineQualityModel") \
    .getOrCreate()

# Paths for training and validation datasets
training_dataset_path = "hdfs:///data/training.csv"
validation_dataset_path = "hdfs:///data/validation.csv"

# Load datasets from HDFS
train_data = spark.read.csv(training_dataset_path, header=True, inferSchema=True, sep=";", quote='"')
val_data = spark.read.csv(validation_dataset_path, header=True, inferSchema=True, sep=";", quote='"')

# Clean column names
train_data = train_data.toDF(*[col.strip('"').strip() for col in train_data.columns])
val_data = val_data.toDF(*[col.strip('"').strip() for col in val_data.columns])

# Assemble features
feature_columns = [col for col in train_data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
train_data = assembler.transform(train_data).select("raw_features", "quality")
val_data = assembler.transform(val_data).select("raw_features", "quality")

# Scale features
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("features", "quality")
val_data = scaler_model.transform(val_data).select("features", "quality")

# Initialize Logistic Regression
lr = LogisticRegression(labelCol="quality", featuresCol="features", maxIter=100)

# Hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="f1"),
                          numFolds=2)

# Train model
cv_model = crossval.fit(train_data)

# Validate model
predictions = cv_model.transform(val_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score:.4f}")

# Save model to hdfs
model_path = "hdfs:///models/wine_quality_model"
cv_model.bestModel.write().overwrite().save(model_path)
print(f"Model saved at {model_path}")

spark.stop()
