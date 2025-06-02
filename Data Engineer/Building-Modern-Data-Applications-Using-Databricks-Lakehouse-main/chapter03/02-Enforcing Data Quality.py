# Databricks notebook source
# MAGIC %md
# MAGIC ### Building a DLT pipeline
# MAGIC Let's begin by importing the `dlt` module.
# MAGIC
# MAGIC The DLT Python module contains function decorators for declaring datasets and establishing dependencies.
# MAGIC
# MAGIC

# COMMAND ----------

import dlt
from pyspark.sql.functions import *

# COMMAND ----------

@dlt.table(
  comment="The randomly generated taxi trip dataset"
)
def yellow_taxi_raw():
  path = "/tmp/chp_03/taxi_data"
  schema = "trip_id INT, taxi_number INT, passenger_count INT, trip_amount FLOAT, trip_distance FLOAT, trip_date DATE"
  return (spark.readStream
               .schema(schema)
               .format("json")
               .load(path))

# COMMAND ----------

@dlt.table(name="trip_data_financials",
           comment="Financial information from incoming taxi trips.")
@dlt.expect("valid_total_amount", "trip_amount > 0.0")
def trip_data_financials():
  return (dlt.readStream("yellow_taxi_raw")
             .withColumn("driver_payment", expr("trip_amount * 0.40"))
             .withColumn("vehicle_maintenance_fee", expr("trip_amount * 0.05"))
             .withColumn("adminstrative_fee", expr("trip_amount * 0.1"))
             .withColumn("potential_profits", expr("trip_amount * 0.45")))

# COMMAND ----------

assertions = {
   "total_amount_constraint": "trip_amount > 0.0",
   "passenger_count": "passenger_count >= 1"
}

@dlt.table(
   name="yellow_taxi_validated",
   comment="A dataset containing trip data that has been validated.")
@dlt.expect_all_or_drop(assertions)
def yellow_taxi_validated():
   return (dlt.readStream("yellow_taxi_raw")
      .withColumn("nyc_congestion_tax", expr("trip_amount * 0.05"))) 
