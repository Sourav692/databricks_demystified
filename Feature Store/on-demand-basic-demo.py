# Databricks notebook source
# MAGIC %md
# MAGIC # On-demand features - basic demo
# MAGIC
# MAGIC This example trains and scores a model that uses an on-demand feature. 
# MAGIC
# MAGIC The feature parses a JSON string to extract a list of hover times on a webpage. These times are averaged together, and the mean is passed as a feature to a model.
# MAGIC
# MAGIC Requirements:
# MAGIC * A cluster running Databricks Runtime for ML 13.3 LTS or above.
# MAGIC * The cluster access model must be Single user.

# COMMAND ----------

# MAGIC %md ## Helper functions and notebook variables

# COMMAND ----------

import json
import random
import time

import pandas as pd
import requests
from datetime import datetime
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
from pyspark.sql.types import IntegerType, BooleanType, StructField, StructType, StringType

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

fe = FeatureEngineeringClient()
suffix = random.randint(1, 10000000000)
registered_model_name = f"main.on_demand_demo.simple_model_{suffix}"
endpoint_name = f"on_demand_demo_simple_endpoint_{suffix}"
should_cleanup = True

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a schema to store the demo feature tables and functions
# MAGIC CREATE SCHEMA IF NOT EXISTS main.on_demand_demo

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Define the Python UDF
# MAGIC CREATE OR REPLACE FUNCTION main.on_demand_demo.avg_hover_time(blob STRING)
# MAGIC RETURNS FLOAT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Extract hover time from JSON blob and computes average"
# MAGIC AS $$
# MAGIC import json
# MAGIC
# MAGIC def calculate_average_hover_time(json_blob):
# MAGIC     # Parse the JSON blob
# MAGIC     data = json.loads(json_blob)
# MAGIC
# MAGIC     # Ensure the 'hover_time' list exists and is not empty
# MAGIC     hover_time_list = data.get('hover_time')
# MAGIC     if not hover_time_list:
# MAGIC         raise ValueError("No hover_time list found or list is empty")
# MAGIC
# MAGIC     # Sum the hover time durations and calculate the average
# MAGIC     total_duration = sum(hover_time_list)
# MAGIC     average_duration = total_duration / len(hover_time_list)
# MAGIC
# MAGIC     return average_duration
# MAGIC
# MAGIC return calculate_average_hover_time(blob)
# MAGIC $$

# COMMAND ----------

# MAGIC %md You can call the Python UDF from SQL, as shown in the next cell.

# COMMAND ----------

# MAGIC %sql SELECT main.on_demand_demo.avg_hover_time('{"hover_time": [5.0, 3.2, 4.1, 2.8, 6.7]}')

# COMMAND ----------

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("json_blob", StringType(), True),
    StructField("label", BooleanType(), True)
])
data = [
  (1, '{"hover_time": [3.5, 4.2, 5.1, 2.8, 3.3]}', True), 
  (2, '{"hover_time": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7]}', False), 
  (3, '{"hover_time": [7.8, 8.9, 6.1, 4.0, 5.3]}', True)
]

label_df = spark.createDataFrame(data, schema)

# COMMAND ----------

# MAGIC %md ## Create a TrainingSet with on-demand features

# COMMAND ----------

features = [
    FeatureFunction(
        udf_name="main.on_demand_demo.avg_hover_time",
        output_name="on_demand_output",
        input_bindings={"blob": "json_blob"},
    )
]

training_set = fe.create_training_set(
    df=label_df, feature_lookups=features, label="label", exclude_columns=["id"]
)

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

# MAGIC %md ## Log a simple model using the TrainingSet
# MAGIC
# MAGIC For simplicity, this notebook uses a hard-coded model. In practice, you'll log a model trained on the generated TrainingSet.

# COMMAND ----------

class HighViewTime(mlflow.pyfunc.PythonModel):
    def predict(self, ctx, inp):
        return inp['on_demand_output'] > 5

# COMMAND ----------

model_name = "fs_packaged_model"
mlflow.set_registry_uri("databricks-uc")

fe.log_model(
    model=HighViewTime(),
    artifact_path=model_name,
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=registered_model_name
)

# COMMAND ----------

# MAGIC %md ## Score the model using score_batch

# COMMAND ----------

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("json_blob", StringType(), True),
])
data = [
  (4, '{"hover_time": [2.1, 3.1, 4.1, 5.1, 6.1]}'), 
  (5, '{"hover_time": [4.4, 5.5, 6.6, 7.7, 8.8]}'), 
]

scoring_df = spark.createDataFrame(data, schema)

# COMMAND ----------

result = fe.score_batch( 
  model_uri = f"models:/{registered_model_name}/1",
  df = scoring_df,
  result_type = 'bool'
)

display(result)

# COMMAND ----------

# MAGIC %md ## Serve the Feature Store packaged model

# COMMAND ----------

start_endpoint_json_body = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": registered_model_name,
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    },
}

host_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
headers = {
  "Authorization": f"Bearer {host_creds.token}",
  "Content-Type": "application/json"
}

response = requests.request(
    url=f"{host_creds.host}/api/2.0/serving-endpoints",
    method="POST",
    json=start_endpoint_json_body,
    headers=headers
)

assert (
    response.status_code == 200
), f"Failed to launch model serving cluster: {response.text}"

print("Starting model serving endpoint. See Serving page for status.")

# COMMAND ----------

# MAGIC %md Wait for the model serving endpoint to be ready.

# COMMAND ----------

model_serving_endpoint_ready = False
num_seconds_per_attempt = 15
num_attempts = 100
for attempt_num in range(num_attempts):
    print(f"Waiting for model serving endpoint {endpoint_name} to be ready...")
    time.sleep(num_seconds_per_attempt)
    response = requests.request(
        url=f"{host_creds.host}/api/2.0/preview/serving-endpoints/{endpoint_name}",
        method="GET",
        headers=headers
    )
    json_response = response.json()
    if (
        response.json()["state"]["ready"] == "READY"
        and response.json()["state"]["config_update"] == "NOT_UPDATING"
    ):
        model_serving_endpoint_ready = True
        break

assert(model_serving_endpoint_ready), f"Model serving endpoint {endpoint_name} not ready after {(num_seconds_per_attempt * num_attempts) / 60} minutes"

# COMMAND ----------

# MAGIC %md ## Query the endpoint

# COMMAND ----------

req = pd.DataFrame(
    [
        {"json_blob": '{"hover_time": [5.5, 2.3, 10.3]}'}
    ]
)
json_req = json.dumps({"dataframe_split": json.loads(req.to_json(orient="split"))})

response = requests.request(
  method="POST", 
  headers=headers, 
  url=f"{host_creds.host}/serving-endpoints/{endpoint_name}/invocations", 
  data=json_req
)

response.json()

# COMMAND ----------

# MAGIC %md Alternatively, use the Serving query endpoints UI to send a request:
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"json_blob": "{\"hover_time\": [5.5, 2.3, 10.3]}"}
# MAGIC   ]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Cleanup

# COMMAND ----------

if should_cleanup:
  MlflowClient().delete_registered_model(name=registered_model_name)
  requests.request(
    method="DELETE", 
    headers=headers, 
    url=f"{host_creds.host}/api/2.0/preview/serving-endpoints/{endpoint_name}" 
  )
