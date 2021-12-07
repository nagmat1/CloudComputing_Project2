from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import round
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.sql import SparkSession
import numpy as np
from sklearn import metrics


spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .appName('CloudComputing_MLlib') \
    .getOrCreate()

df = spark.read.option("delimiter",';').csv('winequality-white.csv', inferSchema=True, header=True)


hasher = FeatureHasher(inputCols=[c for c in df.columns if c not in {'quality'}], outputCol="features")
featurized = hasher.transform(df)
train, test = featurized.randomSplit([0.8, 0.2], seed=57104)

rf = RandomForestRegressor(featuresCol='features', labelCol='quality', predictionCol='pred_quality', maxDepth=30)
pipeline = Pipeline(stages=[rf])
rf_model = pipeline.fit(train)
predictions = rf_model.transform(test)
predictions.select("pred_quality","quality").show()

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 10]).build()
crossval = CrossValidator(estimator=pipeline,estimatorParamMaps=paramGrid,evaluator=RegressionEvaluator().setLabelCol("quality").setPredictionCol("pred_quality"),numFolds=2)
cvModel = crossval.fit(train)
print("cvModel = ",cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
predictions = cvModel.transform(test)
predictions.select("pred_quality","quality").show()
evaluator = RegressionEvaluator(labelCol="quality",predictionCol="pred_quality",metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(" Root Mean Squared Error (RMSE) on test data = {}".format(rmse))
