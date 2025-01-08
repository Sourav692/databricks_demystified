# Databricks notebook source
# MAGIC %md
# MAGIC # On-demand features - Restaurant recommendation demo
# MAGIC
# MAGIC In this example, a restaurant recommendation model takes a JSON strong containing a user's location and a restaurant id. The restaurant's location is looked up from a pre-materialized feature table published to an online store, and an on-demand feature computes the distance from the user to the restaurant. This distance is passed as input to a model.
# MAGIC
# MAGIC Requirements:
# MAGIC * A cluster running Databricks Runtime for ML 13.3 LTS or above.
# MAGIC * The cluster access model must be Single user.

# COMMAND ----------

# MAGIC %md ## Helper functions and notebook variables

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install mlflow>=2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import random
import time

import pandas as pd
import requests
from datetime import datetime
from databricks.feature_engineering import (FeatureFunction, FeatureEngineeringClient, FeatureLookup)
from pyspark.sql.types import IntegerType, BooleanType, StructField, StructType, StringType, TimestampType, FloatType

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

fe = FeatureEngineeringClient()
suffix = random.randint(1, 10000000000)
registered_model_name = f"main.on_demand_demo.restaurant_model_{suffix}"
endpoint_name = f"on_demand_demo_restaurant_endpoint_{suffix}"
should_cleanup = True

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a schema to store the demo feature tables and functions.
# MAGIC CREATE SCHEMA IF NOT EXISTS main.on_demand_demo

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION main.on_demand_demo.extract_user_latitude(blob STRING)
# MAGIC RETURNS FLOAT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Extract latitude from a JSON blob"
# MAGIC AS $$
# MAGIC import json
# MAGIC
# MAGIC def extract_latitude(json_blob: str):
# MAGIC     # Parse the JSON blob
# MAGIC     data = json.loads(json_blob)
# MAGIC     x = data.get('user_x_coord')
# MAGIC
# MAGIC     return x
# MAGIC
# MAGIC return extract_latitude(blob)
# MAGIC $$;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION main.on_demand_demo.extract_user_longitude(blob STRING)
# MAGIC RETURNS FLOAT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Extract longitude from a JSON blob"
# MAGIC AS $$
# MAGIC import json
# MAGIC
# MAGIC def extract_longitude(json_blob: str):
# MAGIC     # Parse the JSON blob
# MAGIC     data = json.loads(json_blob)
# MAGIC     y = data.get('user_y_coord')
# MAGIC
# MAGIC     return y
# MAGIC
# MAGIC return extract_longitude(blob)
# MAGIC $$;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION main.on_demand_demo.haversine_distance(x1 FLOAT, y1 FLOAT, x2 FLOAT, y2 FLOAT)
# MAGIC RETURNS FLOAT
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT "Computes the haversine distance between two points"
# MAGIC AS $$
# MAGIC import math
# MAGIC
# MAGIC def haversine(x1, y1, x2, y2):
# MAGIC     # Radius of the Earth in km
# MAGIC     R = 6371.0
# MAGIC
# MAGIC     # Convert latitude and longitude from degrees to radians
# MAGIC     x1, y1, x2, y2 = map(math.radians, [x1, y1, x2, y2])
# MAGIC
# MAGIC     # Differences in coordinates
# MAGIC     dlat = x2 - x1
# MAGIC     dlon = y2 - y1
# MAGIC
# MAGIC     # Haversine formula
# MAGIC     a = math.sin(dlat / 2)**2 + math.cos(x1) * math.cos(x2) * math.sin(dlon / 2)**2
# MAGIC     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
# MAGIC
# MAGIC     # Distance in kilometers
# MAGIC     distance = R * c
# MAGIC
# MAGIC     return distance
# MAGIC
# MAGIC return haversine(x1, y1, x2, y2)
# MAGIC $$;

# COMMAND ----------

# MAGIC %md You can call the Python UDF from SQL, as shown in the next cell.

# COMMAND ----------

# MAGIC %sql SELECT main.on_demand_demo.extract_user_longitude('{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}')

# COMMAND ----------

# MAGIC %sql SELECT main.on_demand_demo.haversine_distance(37.79122896768446, -122.39362610820227, 37.38703669670432, -122.06163751824737)
# MAGIC

# COMMAND ----------

# MAGIC %md ### Generate and publish feature data

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS main.on_demand_demo.restaurant_features(
# MAGIC   restaurant_id INT NOT NULL,
# MAGIC   ts TIMESTAMP NOT NULL,
# MAGIC   latitude float,
# MAGIC   longitude float,
# MAGIC   CONSTRAINT restaurant_features_pk PRIMARY KEY (restaurant_id, ts TIMESERIES)
# MAGIC )
# MAGIC COMMENT "Restaurant features";

# COMMAND ----------

# Write data to feature table
schema = StructType([
    StructField("restaurant_id", IntegerType(), False),
    StructField("ts", TimestampType(), False),
    StructField("latitude", FloatType(), True),
    StructField("longitude", FloatType(), True),
])

# Create some dummy data
data = [(1, datetime(2023, 9, 25, 12, 0, 0), 40.7128, -74.0060),
        (2, datetime(2023, 9, 25, 13, 0, 0), 34.0522, -118.2437),
        (3, datetime(2023, 9, 25, 14, 0, 0), 41.8781, -87.6298)]

# Create a DataFrame with the dummy data
df = spark.createDataFrame(data, schema=schema)

fe.write_table(name="main.on_demand_demo.restaurant_features", df=df)

# COMMAND ----------

# MAGIC %md To access the feature table from Model Serving, you must create an online table.  
# MAGIC

# COMMAND ----------

# MAGIC %md ## Set up Databricks Online Tables
# MAGIC
# MAGIC You can create an online table from the Catalog Explorer UI, Databricks SDK or Rest API. The steps to use Databricks python SDK are described below. For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#create)). For information about required permissions, see Permissions ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).

# COMMAND ----------

from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
import mlflow

online_table_name = f"main.on_demand_demo.restaurant_features_online"

workspace = WorkspaceClient()

# Create an online table
spec = OnlineTableSpec(
  primary_key_columns = ["restaurant_id"],
  timeseries_key = "ts",
  source_table_full_name = f"main.on_demand_demo.restaurant_features",
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
  perform_full_copy=True)

try:
  online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)
except Exception as e:
  if "already exists" in str(e):
    print(f"Online table {online_table_name} already exists. Not recreating.")  
  else:
    raise e

pprint(workspace.online_tables.get(online_table_name))

# COMMAND ----------

# MAGIC %md ## Create a TrainingSet with on-demand features

# COMMAND ----------

schema = StructType([
    StructField("restaurant_id", IntegerType(), True),
    StructField("json_blob", StringType(), True),
    StructField("ts", TimestampType(), False),
    StructField("label", BooleanType(), True)
])
data = [
  (2, '{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}', datetime(2023, 9, 26, 12, 0, 0), True), 
]

label_df = spark.createDataFrame(data, schema)

# COMMAND ----------

features = [
    FeatureLookup(
        table_name = "main.on_demand_demo.restaurant_features",
        feature_names = ["latitude", "longitude"],
        rename_outputs={"latitude": "restaurant_latitude", "longitude": "restaurant_longitude"},
        lookup_key = "restaurant_id",
        timestamp_lookup_key = "ts"
    ),
    FeatureFunction(
        udf_name="main.on_demand_demo.extract_user_latitude",
        output_name="user_latitude",
        input_bindings={"blob": "json_blob"},
    ),
    FeatureFunction(
        udf_name="main.on_demand_demo.extract_user_longitude",
        output_name="user_longitude",
        input_bindings={"blob": "json_blob"},
    ),    
    FeatureFunction(
        udf_name="main.on_demand_demo.haversine_distance",
        output_name="distance",
        input_bindings={"x1": "restaurant_longitude", "y1": "restaurant_latitude", "x2": "user_longitude", "y2": "user_latitude"},
    )

]

training_set = fe.create_training_set(
    df=label_df, feature_lookups=features, label="label", exclude_columns=["restaurant_id", "json_blob", "restaurant_latitude", "restaurant_longitude", "user_latitude", "user_longitude", "ts"]
)

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

# MAGIC %md ## Log a simple model using the TrainingSet
# MAGIC
# MAGIC For simplicity, this notebook uses a hard-coded model. In practice, you'll log a model trained on the generated TrainingSet.

# COMMAND ----------

class IsClose(mlflow.pyfunc.PythonModel):
    def predict(self, ctx, inp):
        return (inp['distance'] < 2.5).values

# COMMAND ----------

model_name = "fs_packaged_model"
mlflow.set_registry_uri("databricks-uc")

fe.log_model(
    model=IsClose(),
    artifact_path=model_name,
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=registered_model_name
)

# COMMAND ----------

# MAGIC %md ## Score the model using score_batch

# COMMAND ----------

schema = StructType([
    StructField("restaurant_id", IntegerType(), True),
    StructField("json_blob", StringType(), True),
    StructField("ts", TimestampType(), False),
])
data = [
  (2, '{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}', datetime(2023, 9, 26, 12, 0, 0)), 
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

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import datetime

workspace = WorkspaceClient()

# Create endpoint
workspace.serving_endpoints.create_and_wait(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
    ServedEntityInput(
        entity_name=registered_model_name,
        entity_version="1",
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  ),
  timeout=datetime.timedelta(minutes=30),
)

# COMMAND ----------

# MAGIC %md ## Query the endpoint

# COMMAND ----------

host_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
headers = {
  "Authorization": f"Bearer {host_creds.token}",
  "Content-Type": "application/json"
}

req = pd.DataFrame(
    [
        {
          "restaurant_id": 2,
          "json_blob": '{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}',
        }
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
# MAGIC     {
# MAGIC           "restaurant_id": 2,
# MAGIC           "json_blob": "{\"user_x_coord\": 37.79122896768446, \"user_y_coord\": -122.39362610820227}"   
# MAGIC     }
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
