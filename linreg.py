from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import round
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("CloudComputing_MLlib").getOrCreate()
df = spark.read.option("delimiter",';').csv('winequality-white.csv', inferSchema=True, header=True)

features = df.columns
X_feat = features.remove('quality')

hasher = FeatureHasher(inputCols=[c for c in df.columns if c not in {'quality'}], outputCol="features")
featurized = hasher.transform(df)
train, test = featurized.randomSplit([0.8, 0.2], seed=57104)

lr = LinearRegression(maxIter=1250, regParam=0.03, elasticNetParam=0.8).setLabelCol("quality")
# Fit the model
lrModel = lr.fit(train)

predictions = lrModel.transform(test)
predictions.select("prediction", "quality").show(10)
predictions = lrModel.transform(test)

evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="r2")
r_square = evaluator.evaluate(predictions)
print(" R^2 on test data for LINEAR REGRESSION = {}".format(r_square))


predictionAndLabels=predictions.withColumn("prediction", round(predictions["prediction"]).cast(DoubleType())).select("prediction", "quality")
predictionAndLabels.show(10)
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
f1 = evaluatorMulti.evaluate(predictionAndLabels, {evaluatorMulti.metricName: "f1"})

print("F1 score = ",f1)


