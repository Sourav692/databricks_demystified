# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# The following line automatically generates a PAT Token for authentication
client = VectorSearchClient()

# The following line uses the service principal token for authentication
# client = VectorSearch(service_principal_client_id=<CLIENT_ID>,service_principal_client_secret=<CLIENT_SECRET>)

client.create_endpoint(
    name="vector_search_endpoint_sourav",
    endpoint_type="STANDARD"
)

# COMMAND ----------


