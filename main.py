"""Create spark context"""
import pyspark

# spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
#             .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3") \
#             .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
#             .getOrCreate()
spark = pyspark.sql.SparkSession.builder.appName("MMLsparkTest").getOrCreate()


"""Make mmlspark available"""
import sys

sys.path.append("/home/renxiang/Developer/mmlspark_sandbox/mmlspark_2.11-1.0.0-rc1.jar")

"""Rest of the script"""
import numpy as np
import pandas as pd

dataFile = "AdultCensusIncome.csv"
import os
import urllib.request


if not os.path.isfile(dataFile):
    urllib.request.urlretrieve(
        "https://mmlspark.azureedge.net/datasets/" + dataFile, dataFile
    )
data = spark.createDataFrame(
    pd.read_csv(dataFile, dtype={" hours-per-week": np.float64})
)
data.show(5)

data = data.select([" education", " marital-status", " hours-per-week", " income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)

from mmlspark.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression

model = TrainClassifier(model=LogisticRegression(), labelCol=" income").fit(train)

from mmlspark.train import ComputeModelStatistics

prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
metrics.select("accuracy").show()
