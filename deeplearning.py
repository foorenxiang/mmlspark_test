import sys

sys.path.append("mmlspark_2.11-1.0.0-rc1.jar")


import pyspark

# spark = pyspark.sql.SparkSession.builder.appName("MyApp").getOrCreate()

spark = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    # .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3")
    # .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    .getOrCreate()
)


import mmlspark

mmlspark


from pyspark.ml import Transformer, Estimator, Pipeline
from pyspark.ml.classification import LogisticRegression
from mmlspark.downloader import ModelDownloader
import os, time


model = ModelDownloader(spark, "dbfs:/models/").downloadByName("ResNet50")


# Load the images
# use flowers_and_labels.parquet on larger cluster in order to get better results
imagesWithLabels = (
    spark.read.parquet(
        "wasbs://publicwasb@mmlspark.blob.core.windows.net/flowers_and_labels2.parquet"
    )
    .withColumnRenamed("bytes", "image")
    .sample(0.1)
)

imagesWithLabels.printSchema()


from mmlspark.opencv import ImageTransformer
from mmlspark.image import UnrollImage
from mmlspark.cntk import ImageFeaturizer
from mmlspark.stages import *

# Make some featurizers
it = ImageTransformer().setOutputCol("scaled").resize(size=(60, 60))

ur = UnrollImage().setInputCol("scaled").setOutputCol("features")

dc1 = DropColumns().setCols(["scaled", "image"])

lr1 = (
    LogisticRegression().setMaxIter(8).setFeaturesCol("features").setLabelCol("labels")
)

dc2 = DropColumns().setCols(["features"])

basicModel = Pipeline(stages=[it, ur, dc1, lr1, dc2])


resnet = (
    ImageFeaturizer()
    .setInputCol("image")
    .setOutputCol("features")
    .setModelLocation(model.uri)
    .setLayerNames(model.layerNames)
    .setCutOutputLayers(1)
)

dc3 = DropColumns().setCols(["image"])

lr2 = (
    LogisticRegression().setMaxIter(8).setFeaturesCol("features").setLabelCol("labels")
)

dc4 = DropColumns().setCols(["features"])

deepModel = Pipeline(stages=[resnet, dc3, lr2, dc4])


def timedExperiment(model, train, test):
    start = time.time()
    result = model.fit(train).transform(test).toPandas()
    print("Experiment took {}s".format(time.time() - start))
    return result


train, test = imagesWithLabels.randomSplit([0.8, 0.2])
train.count(), test.count()


basicResults = timedExperiment(basicModel, train, test)


deepResults = timedExperiment(deepModel, train, test)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate(results, name):
    y, y_hat = results["labels"], results["prediction"]
    y = [int(l) for l in y]

    accuracy = np.mean([1.0 if pred == true else 0.0 for (pred, true) in zip(y_hat, y)])
    cm = confusion_matrix(y, y_hat)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.text(
        40, 10, "$Accuracy$ $=$ ${}\%$".format(round(accuracy * 100, 1)), fontsize=14
    )
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("$Predicted$ $label$", fontsize=18)
    plt.ylabel("$True$ $Label$", fontsize=18)
    plt.title("$Normalized$ $CM$ $for$ ${}$".format(name))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
evaluate(deepResults, "CNTKModel + LR")
plt.subplot(1, 2, 2)
evaluate(basicResults, "LR")
# Note that on the larger dataset the accuracy will bump up from 44% to >90%
display(plt.show())
