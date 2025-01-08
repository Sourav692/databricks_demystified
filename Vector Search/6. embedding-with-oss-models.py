# Databricks notebook source
# MAGIC %md # Register and serve an OSS embedding model
# MAGIC
# MAGIC This notebook sets up the open source text embedding model `e5-small-v2` in a Model Serving endpoint usable for Vector Search.
# MAGIC * Download the model from the Hugging Face Hub.
# MAGIC * Register it to the MLflow Model Registry.
# MAGIC * Start a Model Serving endpoint to serve the model.
# MAGIC
# MAGIC The model `e5-small-v2` is available at https://huggingface.co/intfloat/e5-small-v2.
# MAGIC * MIT license
# MAGIC * Variants:
# MAGIC    * https://huggingface.co/intfloat/e5-large-v2
# MAGIC    * https://huggingface.co/intfloat/e5-base-v2
# MAGIC    * https://huggingface.co/intfloat/e5-small-v2
# MAGIC
# MAGIC For a list of library versions included in Databricks Runtime, see the release notes for your Databricks Runtime version ([AWS](https://docs.databricks.com/en/release-notes/runtime/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/release-notes/runtime/index/)).

# COMMAND ----------

# MAGIC %md ## Install Databricks Python SDK
# MAGIC
# MAGIC This notebook uses its Python client to work with serving endpoints.

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk python-snappy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

!pip install -U sentence-transformers

# COMMAND ----------

# MAGIC %md ## Download model

# COMMAND ----------

# Download model using the sentence_transformers library.
from sentence_transformers import SentenceTransformer

source_model_name = 'intfloat/e5-small-v2'  # model name on Hugging Face Hub
model = SentenceTransformer(source_model_name)

# COMMAND ----------

# Test the model, just to show it works.
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

# COMMAND ----------

# MAGIC %md ## Register model to MLflow

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------


# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.
catalog = "main"
schema = "default"
model_name = "e5-small-v2"

# COMMAND ----------

# MLflow model name. The Model Registry uses this name for the model.
registered_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

# Compute input and output schema.
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

model_info = mlflow.sentence_transformers.log_model(
  model,
  artifact_path="model",
  signature=signature,
  input_example=sentences,
  registered_model_name=registered_model_name)

# COMMAND ----------

inference_test = ["I enjoy pies of both apple and cherry.", "I prefer cookies."]

# Load the custom model by providing the URI for where the model was logged.
loaded_model_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

# Perform a quick test to ensure that the loaded model generates the correct output.
embeddings_test = loaded_model_pyfunc.predict(inference_test)
embeddings_test

# COMMAND ----------

# Extract the version of the model you just registered.
mlflow_client = mlflow.MlflowClient()

def get_latest_model_version(model_name):
  client = mlflow_client
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_version = get_latest_model_version(registered_model_name)
model_version

# COMMAND ----------

import requests

API_URL = "https://api-inference.huggingface.co/models/intfloat/e5-small-v2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
	"source_sentence": "That is a happy person",
	"sentences": [
		"That is a happy dog",
		"That is a very happy person",
		"Today is a sunny day"
	]
},
})

# COMMAND ----------

output

# COMMAND ----------

# MAGIC %md ### Create model serving endpoint
# MAGIC
# MAGIC For more details, see "Create foundation model serving endpoints" ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-foundation-model-endpoints/)).
# MAGIC
# MAGIC **Note**: This example creates a *small* CPU endpoint that scales down to 0.  This is for quick, small tests. For more realistic use cases, consider using GPU endpoints for faster embedding computation and not scaling down to 0 if you expect frequent queries, as Model Serving endpoints have some cold start overhead.

# COMMAND ----------

endpoint_name = "e5-small-v2"  # Name of endpoint to create

# COMMAND ----------

# DBTITLE 1,Create Databricks SDK workspace client
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,Create endpoint
endpoint_config_dict = {
    "served_entities": [
        {
            "name": f'{registered_model_name.replace(".", "_")}_{1}',
            "entity_name": registered_model_name,
            "entity_version": model_version,
            "workload_type": "CPU",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }
    ]
}


endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# The endpoint may take several minutes to get ready.
w.serving_endpoints.create_and_wait(name=endpoint_name, config=endpoint_config)

# COMMAND ----------

# MAGIC %md ### Query endpoint
# MAGIC
# MAGIC The above `create_and_wait` command waits until the endpoint is ready. You can also check the status of the serving endpoint in the Databricks UI.
# MAGIC
# MAGIC For more information, see "Query foundation models" ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-model-serving-endpoints/)).

# COMMAND ----------

# Only run this command after the Model Serving endpoint is in the Ready state.
import time

start = time.time()

# If the endpoint is not yet ready, you might get a timeout error. If so, wait and then rerun the command.
endpoint_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=['Hello world', 'Good morning'])

end = time.time()

print(endpoint_response)
print(f'Time taken for querying endpoint in seconds: {end-start}')

# COMMAND ----------


