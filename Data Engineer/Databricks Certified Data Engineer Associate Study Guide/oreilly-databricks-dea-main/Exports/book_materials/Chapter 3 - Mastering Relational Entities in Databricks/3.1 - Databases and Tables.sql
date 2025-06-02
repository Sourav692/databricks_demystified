-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Putting Relational Entities into Practice

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Working in the Default Schema

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating managed tables

-- COMMAND ----------

USE CATALOG hive_metastore;

CREATE TABLE managed_default
 (country STRING, code STRING, dial_code STRING);

INSERT INTO managed_default
VALUES ('France', 'Fr', '+33')

-- COMMAND ----------

DESCRIBE EXTENDED managed_default

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating external tables

-- COMMAND ----------

CREATE TABLE external_default
 (country STRING, code STRING, dial_code STRING)
LOCATION 'dbfs:/mnt/demo/external_default';

INSERT INTO external_default
VALUES ('France', 'Fr', '+33')

-- COMMAND ----------

DESCRIBE EXTENDED external_default

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Dropping tables

-- COMMAND ----------

DROP TABLE managed_default

-- COMMAND ----------

SELECT * FROM managed_default

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/managed_default'

-- COMMAND ----------

DROP TABLE external_default

-- COMMAND ----------

SELECT * FROM external_default

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/mnt/demo/external_default'

-- COMMAND ----------

SELECT * FROM DELTA.`dbfs:/mnt/demo/external_default`

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.rm('dbfs:/mnt/demo/external_default', True)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Working in a New Schema

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating a new database

-- COMMAND ----------

CREATE SCHEMA new_default

-- COMMAND ----------

DESCRIBE DATABASE EXTENDED new_default

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating tables in the new database

-- COMMAND ----------

USE DATABASE new_default;

-- create a managed table
CREATE TABLE managed_new_default
 (country STRING, code STRING, dial_code STRING);

INSERT INTO managed_new_default
VALUES ('France', 'Fr', '+33');
-----------------------------------
-- Create an external table
CREATE TABLE external_new_default
 (country STRING, code STRING, dial_code STRING)
LOCATION 'dbfs:/mnt/demo/external_new_default';

INSERT INTO external_new_default
VALUES ('France', 'Fr', '+33');

-- COMMAND ----------

DESCRIBE EXTENDED managed_new_default

-- COMMAND ----------

DESCRIBE EXTENDED external_new_default

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Dropping tables

-- COMMAND ----------

DROP TABLE managed_new_default;
DROP TABLE external_new_default;

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/new_default.db/managed_new_default'

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/mnt/demo/external_new_default'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Working In a Custom-Location Schema

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating the database

-- COMMAND ----------

CREATE SCHEMA custom
LOCATION 'dbfs:/Shared/schemas/custom.db'

-- COMMAND ----------

DESCRIBE DATABASE EXTENDED custom

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating tables

-- COMMAND ----------

USE DATABASE custom;

-- Create a managed table
CREATE TABLE managed_custom
 (country STRING, code STRING, dial_code STRING);

INSERT INTO managed_custom
VALUES ('France', 'Fr', '+33');
-----------------------------------
-- Create an external table
CREATE TABLE external_custom
 (country STRING, code STRING, dial_code STRING)
LOCATION 'dbfs:/mnt/demo/external_custom';

INSERT INTO external_custom
VALUES ('France', 'Fr', '+33');

-- COMMAND ----------

DESCRIBE EXTENDED managed_custom

-- COMMAND ----------

DESCRIBE EXTENDED external_custom

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Dropping tables

-- COMMAND ----------

DROP TABLE managed_custom;
DROP TABLE external_custom;

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/Shared/schemas/custom.db/managed_custom'

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/mnt/demo/external_custom'
