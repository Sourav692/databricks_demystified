-- Databricks notebook source
-- MAGIC %md
-- MAGIC #Vector Search System Tables Queries
-- MAGIC
-- MAGIC Databricks Vector Search billing and audit information in Unity Catalog System Tables.  The following queries show how to obtain this information 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Billing Queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Retrieving "search" usage (DBUs) per endpoint 
-- MAGIC - This query uses the ``system.billing.usage`` table to retrieve the number of dbus used per Vector Search endpoint in the last 30 days 

-- COMMAND ----------

select
  usage_metadata.endpoint_name as endpoint_name, usage_quantity as dbus, sku_name
from
  system.billing.usage
where
  billing_origin_product = 'VECTOR_SEARCH'
  and usage_metadata.endpoint_name is not null 
  and usage_date between date_add(current_date(), -30) and current_date()
  -- and sku_name like "ENTERPRISE_SERVERLESS_REAL_TIME_INFERENCE_%" 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Retrieving "ingestion" usage (DBUs) per DLT pipeline 
-- MAGIC - This query uses the ``system.billing.usage`` table to retrieve the number of dbus used per Ingestion DLT pipeline id in the last 30 days 

-- COMMAND ----------

select
  usage_metadata.dlt_pipeline_id as dlt_pipeline, 
  usage_quantity as dbus,
  sku_name,
  usage_metadata
from
  system.billing.usage
where
  billing_origin_product = 'VECTOR_SEARCH'
  and usage_date between date_add(current_date(), -30) and current_date()
  and usage_metadata.dlt_pipeline_id is not null

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##Audit Queries

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Action audited
-- MAGIC - The following query show the Vector Search actions audited

-- COMMAND ----------

select
 distinct action_name
from
  system.access.audit
where 
  service_name = "vectorSearch" 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Create Endpoint Action
-- MAGIC - This query retrieves the ``endpoint creation`` events of the last 30 days

-- COMMAND ----------

select
 request_params.name as endpoint_name,
 request_params.endpoint_type as endpoint_type,
 *
from
  system.access.audit
where 
  service_name = "vectorSearch"
  and action_name = "createEndpoint"
  and event_date between date_add(current_date(), -30) and current_date()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Delete Endpoint Action
-- MAGIC - This query retrieves the ``endpoint deletion`` events of the last 30 days

-- COMMAND ----------

select
 request_params.name as endpoint_name,
 *
from
  system.access.audit
where 
  service_name = "vectorSearch"
  and action_name = "deleteEndpoint"
  and event_date between date_add(current_date(), -30) and current_date()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Create Index Action
-- MAGIC - This query retrieves the ``index creation`` events of the last 30 days

-- COMMAND ----------

select
  request_params.name as index_name,
  request_params.endpoint_name as endpoint_name,
  request_params.primary_key as primary_key,
  request_params.index_type as index_type,
  request_params.delta_sync_index_spec as delta_sync_index_spec,
  request_params.direct_access_index_spec as direct_access_index_spec,
  *
from
  system.access.audit
where
  service_name = "vectorSearch"
  and action_name = "createVectorIndex"
  and event_date between date_add(current_date(), -30) and current_date()  

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ###Delete Index Action
-- MAGIC - This query retrieves the ``index deletion`` events of the last 30 days

-- COMMAND ----------

select
  request_params.name as index_name,
  request_params.endpoint_name as endpoint_name,
  request_params.delete_embedding_writeback_table as delete_embedding_writeback_table,
  *
from
  system.access.audit
where
  service_name = "vectorSearch"
  and action_name = "deleteVectorIndex"
  and event_date between date_add(current_date(), -30) and current_date()  

-- COMMAND ----------


