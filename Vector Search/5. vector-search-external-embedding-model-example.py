# Databricks notebook source
# MAGIC %md # Vector Search external embedding model (OpenAI) example
# MAGIC
# MAGIC This notebook shows how to use the Vector Search Python SDK, which provides a `VectorSearchClient` as a primary API for working with Vector Search.
# MAGIC
# MAGIC This notebook uses Databricks support of external models ([AWS](https://docs.databricks.com/en/generative-ai/external-models/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/external-models/)) to access  an OpenAI embeddings model to generate embeddings.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# Display help for the Vector Search Client
help(VectorSearchClient)

# COMMAND ----------

# MAGIC %md ## Load toy dataset into source Delta table
# MAGIC
# MAGIC The following creates the source Delta table.

# COMMAND ----------

# Specify the catalog and schema to use. You must have USE_CATALOG privilege on the catalog and USE_SCHEMA and CREATE_TABLE privileges on the schema.
# Change the catalog and schema here if necessary.

catalog_name = "main"
schema_name = "default"

# COMMAND ----------


source_table_name = "wiki_articles_demo"
source_table_fullname = f"{catalog_name}.{schema_name}.{source_table_name}"

# COMMAND ----------

# Uncomment the following line if you want to start from scratch.

# spark.sql(f"DROP TABLE {source_table_fullname}")

# COMMAND ----------

source_df = spark.read.parquet("/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet").limit(10)
display(source_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk sample dataset
# MAGIC
# MAGIC Chunking the sample dataset helps you avoid exceeding the context limit of the embedding model.
# MAGIC The OpenAI model supports up to 8192 tokens. However, Databricks recommends that you split the data into smaller context chunks so that you can feed a wider variety of examples into the reasoning model for your RAG application.
# MAGIC

# COMMAND ----------

import tiktoken
import pandas as pd


max_chunk_tokens = 1024
encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text):
    # Encode and then decode within the UDF
    tokens = encoding.encode(text)
    chunks = []
    while tokens:
        chunk_tokens = tokens[:max_chunk_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        tokens = tokens[max_chunk_tokens:]
    return chunks

# Process the data and store in a new list
pandas_df = source_df.toPandas()
processed_data = []
for index, row in pandas_df.iterrows():
    text_chunks = chunk_text(row['text'])
    chunk_no = 0
    for chunk in text_chunks:
        row_data = row.to_dict()
        
        # Replace the id column with a new unique chunk id
        # and the text column with the text chunk
        row_data['id'] = f"{row['id']}_{chunk_no}"
        row_data['text'] = chunk
        
        processed_data.append(row_data)
        chunk_no += 1

chunked_pandas_df = pd.DataFrame(processed_data)
chunked_spark_df = spark.createDataFrame(chunked_pandas_df)

# Write the chunked DataFrame to a Delta table
spark.sql(f"DROP TABLE IF EXISTS {source_table_fullname}")
chunked_spark_df.write.format("delta") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(source_table_fullname)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {source_table_fullname}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create vector search endpoint

# COMMAND ----------

vector_search_endpoint_name = "vector-search-external-demo-endpoint"

# COMMAND ----------

def endpoint_exists (vsc, endpoint_name):
  if vsc.get_endpoint(name=endpoint_name)["name"] == endpoint_name:
    return True
  else:
    False

# COMMAND ----------

if not endpoint_exists(vsc, vector_search_endpoint_name):
    vsc.create_endpoint(
        name=vector_search_endpoint_name,
        endpoint_type="STANDARD"
    )

# COMMAND ----------

vsc.get_endpoint(
  name=vector_search_endpoint_name
)


# COMMAND ----------

# MAGIC %md ## Register OpenAI embedding model endpoint
# MAGIC
# MAGIC For detailed usage information, see the external model documentation for configuring an OpenAI endpoint ([AWS](https://docs.databricks.com/en/generative-ai/external-models/index.html#configure-the-provider-for-an-endpoint)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/external-models/#--configure-the-provider-for-an-endpoint)).
# MAGIC
# MAGIC To provide credentials, use the Databricks secret manager ([AWS](https://docs.databricks.com/en/security/secrets/example-secret-workflow.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/example-secret-workflow)).

# COMMAND ----------

embedding_model_endpoint_name = "openai-embedding-endpoint_SB"

# COMMAND ----------

import mlflow.deployments

mlflow_deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Configure the secret manager with the OpenAPI key and provide the
# correct scope and key name below.

mlflow_deploy_client.create_endpoint(
    name=embedding_model_endpoint_name,
    config={
        "served_entities": [{
            "external_model": {
                "name": "text-embedding-ada-002",
                "provider": "openai",
                "task": "llm/v1/embeddings",
                "openai_config": {
                    "openai_api_key": "{{secrets/vsc/openai-key}}" # CHANGE ME
                }
            }
    }]
    }
)


# COMMAND ----------

# MAGIC %md ## Create vector index

# COMMAND ----------

# Vector index
vs_index = f"{source_table_name}_openai_index"
vs_index_fullname = f"{catalog_name}.{schema_name}.{vs_index}"

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gtegte-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint=embedding_model_endpoint_name, inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

index = vsc.create_delta_sync_index(
  endpoint_name=vector_search_endpoint_name,
  source_table_name=source_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="text",
  embedding_model_endpoint_name=embedding_model_endpoint_name
)
index.describe()['status']['message']

# COMMAND ----------

# Wait for index to come online. Expect this command to take several minutes.
# You can also track the status of the index build in Catalog Explorer in the
# Overview tab for the vector index.

import time
index = vsc.get_index(endpoint_name=vector_search_endpoint_name,index_name=vs_index_fullname)
while not index.describe().get('status')['ready']:
  print("Waiting for index to be ready...")
  time.sleep(30)
print("Index is ready!")
index.describe()

# COMMAND ----------

# MAGIC %md ## Similarity search
# MAGIC
# MAGIC The following cells show how to query the Vector Index to find similar documents.

# COMMAND ----------

results = index.similarity_search(
  query_text="Greek myths",
  columns=["id", "text", "title"],
  num_results=5
  )
rows = results['result']['data_array']
for (id, text, title, score) in rows:
  if len(text) > 32:
    # trim text output for readability
    text = text[0:32] + "..."
  print(f"id: {id}  title: {title} text: '{text}' score: {score}")

# COMMAND ----------

results = index.similarity_search(
  query_text="Greek myths",
  columns=["id", "text", "title"],
  num_results=5,
  filters={"title NOT": "Hercules"}
)
rows = results['result']['data_array']
for (id, text, title, score) in rows:
  if len(text) > 32:
    # trim text output for readability
    text = text[0:32] + "..."
  print(f"id: {id}  title: {title} text: '{text}' score: {score}")

# COMMAND ----------

# MAGIC %md ## Delete vector index

# COMMAND ----------

vsc.delete_index(
  endpoint_name=vector_search_endpoint_name,
  index_name=vs_index_fullname
)

# COMMAND ----------


