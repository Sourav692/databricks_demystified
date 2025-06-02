-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## View Types

-- COMMAND ----------

USE CATALOG hive_metastore;

SHOW TABLES;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Global temporary views

-- COMMAND ----------

SHOW TABLES IN global_temp;

-- COMMAND ----------

SELECT * FROM global_temp.global_temp_view_recent_cars;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Dropping Views

-- COMMAND ----------

DROP VIEW view_tesla_cars;

-- COMMAND ----------

DROP VIEW global_temp.global_temp_view_recent_cars;

-- COMMAND ----------

DROP TABLE cars
