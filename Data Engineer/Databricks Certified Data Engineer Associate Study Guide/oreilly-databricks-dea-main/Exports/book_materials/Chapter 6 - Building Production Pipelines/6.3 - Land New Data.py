# Databricks notebook source
# MAGIC %md
# MAGIC # Processing Change Data Capture

# COMMAND ----------

# MAGIC %run ../Includes/School-Setup

# COMMAND ----------

load_new_json_data()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from json.`${dataset.school}/courses-cdc/02.json`
