from pdb import pm
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType,StructField,LongType, StringType,DoubleType,TimestampType
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.shell import spark
import time

#define the schema for the input 
schema = StructType([
    StructField("Time",DoubleType(),True), StructField("V1",DoubleType(),True),
    StructField("V2",DoubleType(),True),  StructField("V3",DoubleType(),True),
    StructField("V4",DoubleType(),True), StructField("V5",DoubleType(),True),
    StructField("V6",DoubleType(),True), StructField("V7",DoubleType(),True),
    StructField("V8",DoubleType(),True), StructField("V9",DoubleType(),True),
    StructField("V10",DoubleType(),True), StructField("V11",DoubleType(),True),
    StructField("V12",DoubleType(),True), StructField("V13",DoubleType(),True),
    StructField("V14",DoubleType(),True), StructField("V15",DoubleType(),True),
    StructField("V16",DoubleType(),True), StructField("V17",DoubleType(),True),
    StructField("V18",DoubleType(),True), StructField("V19",DoubleType(),True),
    StructField("V20",DoubleType(),True), StructField("V21",DoubleType(),True),
    StructField("V22",DoubleType(),True), StructField("V23",DoubleType(),True),
    StructField("V24",DoubleType(),True), StructField("V25",DoubleType(),True),
    StructField("V26",DoubleType(),True), StructField("V27",DoubleType(),True),
    StructField("V28",DoubleType(),True), StructField("Amount",DoubleType(),True),
    StructField("Class",LongType(),True)    
])
lr = LogisticRegression(maxIter=10, regParam= 0.01)
#Reading the Csv using Spark.read function - Passing the schema and the filepath
filePath = os.path.join(os.getcwd(), "creditcard.csv")
csvDataFrame=spark.read.format('csv').option('header',True).schema(schema).load(filePath).withColumnRenamed("Class", "label")

csvDataFrame.show()
csvDataFrame.printSchema()

#split the Data into Test and Training Datasets by Random
testDataFrame, trainDataFrame = csvDataFrame.randomSplit([0.3,0.7])

cols_for_assembler = ['V1',"V2","V3","V4","V5","V6","V7", "V8", "V9", "V10",\
                        "V11","V12","V13","V14","V15","V16","V17", "V18", "V19","V20",\
                            "V21","V22","V23","V24","V25","V26","V27", "V28"]

cols_for_scaler = cols_for_assembler.append("Amount")

assembler1 = VectorAssembler(inputCols=cols_for_assembler, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features")
#creating stages for Pipeline
myStages = [assembler1, scaler, lr]

pipeline = Pipeline(stages = myStages)

pModel = pipeline.fit(trainDataFrame)

trainingPred = pModel.transform(trainDataFrame)
trainingPred.select('label', 'probability', 'prediction').show()


testData = testDataFrame.repartition(10)
testData.show()
testFilePath = os.path.join(os.getcwd(), "testData")
print(testFilePath)
testData.write.option("header",True).csv(testFilePath)

#streaming - TestData

sourceStream=spark.readStream.format("csv").option("header",True).schema(schema).option("ignoreLeadingWhiteSpace",True).\
    option("mode","dropMalformed").option("maxFilesPerTrigger",1).load(testFilePath).withColumnRenamed("Class","label")


streamingOutput = pModel.transform(sourceStream).select('label', 'probability', 'prediction')
outputFilePath = os.path.join(os.getcwd(), "testOutputData.csv")
query = streamingOutput.writeStream.format("console").start()
time.sleep(10)
query.stop()

# streamingOutput.writeStream.format("csv").trigger(processingTime="60 seconds").option("checkpointLocation", "checkpoint/")\
#   .option("path", outputFilePath).outputMode("append").start().awaitTermination()


