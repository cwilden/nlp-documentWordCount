// Databricks notebook source
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._


// COMMAND ----------

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline


// COMMAND ----------

val pipeline = PretrainedPipeline("recognize_entities_dl", lang="en")


// COMMAND ----------

val dracula = sc.textFile("/FileStore/tables/dracula.txt")

// COMMAND ----------

val result = pipeline.annotate(dracula.collect())

// COMMAND ----------

val data = dracula.toDF("text")

// COMMAND ----------

val resultDF = pipeline.annotate(data, "text")

// COMMAND ----------

import org.apache.spark.sql.functions._

val sortedNamedE = resultDF.groupBy(col("entities.result")(0) as "entity").count().sort(desc("count"))

// COMMAND ----------

sortedNamedE.show()

// COMMAND ----------

val resultMap = sortedNamedE.rdd.map(row => (row(0) -> row.getLong(1)))

// COMMAND ----------

resultMap.collect()
