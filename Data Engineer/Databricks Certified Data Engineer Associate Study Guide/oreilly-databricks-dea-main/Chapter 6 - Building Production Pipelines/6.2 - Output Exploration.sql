-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Examining DLT pipelines

-- COMMAND ----------

-- MAGIC %fs ls "dbfs:/mnt/DEA-Book/dlt"

-- COMMAND ----------

-- MAGIC %fs ls "dbfs:/mnt/DEA-Book/dlt/system/events"

-- COMMAND ----------

SELECT * FROM delta.`dbfs:/mnt/DEA-Book/dlt/system/events`

-- COMMAND ----------

-- MAGIC %fs ls dbfs:/mnt/DEA-Book/dlt/schemas/school_dlt_db/tables

-- COMMAND ----------

SELECT * FROM hive_metastore.school_dlt_db.uk_daily_student_courses
