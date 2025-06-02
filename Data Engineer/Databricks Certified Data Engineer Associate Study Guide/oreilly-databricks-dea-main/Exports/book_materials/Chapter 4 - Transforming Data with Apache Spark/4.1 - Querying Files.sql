-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Querying Data Files

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div  style="text-align: center;">
-- MAGIC   <img src="https://raw.githubusercontent.com/derar-alhussein/oreilly-databricks-dea/main/Includes/Images/school_schema.png" alt="School Schema">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Querying JSON Format

-- COMMAND ----------

-- MAGIC %run ../Includes/School-Setup

-- COMMAND ----------

-- MAGIC %python
-- MAGIC files = dbutils.fs.ls(f"{dataset_school}/students-json")
-- MAGIC display(files)

-- COMMAND ----------

SELECT * FROM json.`${dataset.school}/students-json/export_001.json`

-- COMMAND ----------

SELECT * FROM json.`${dataset.school}/students-json/export_*.json`

-- COMMAND ----------

SELECT * FROM json.`${dataset.school}/students-json`

-- COMMAND ----------

SELECT COUNT(1) FROM json.`${dataset.school}/students-json`

-- COMMAND ----------

SELECT *, input_file_name() source_file
FROM json.`${dataset.school}/students-json`;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Querying Using the text Format

-- COMMAND ----------

SELECT * FROM text.`${dataset.school}/students-json`

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Querying Using binaryFile Format

-- COMMAND ----------

SELECT * FROM binaryFile.`${dataset.school}/students-json`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Querying Non-Self-Describing Formats

-- COMMAND ----------

SELECT * FROM csv.`${dataset.school}/courses-csv`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Registering Tables from Files with CTAS

-- COMMAND ----------

CREATE TABLE students AS
SELECT * FROM json.`${dataset.school}/students-json`;

DESCRIBE EXTENDED students;

-- COMMAND ----------

CREATE TABLE courses_unparsed AS
SELECT * FROM csv.`${dataset.school}/courses-csv`;

SELECT * FROM courses_unparsed;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Registering Tables on Foreign Data Sources

-- COMMAND ----------

CREATE TABLE courses_csv
 (course_id STRING, title STRING, instructor STRING, category STRING, price DOUBLE)
USING CSV
OPTIONS (
 header = "true",
 delimiter = ";")
LOCATION "${dataset.school}/courses-csv"

-- COMMAND ----------

SELECT * FROM courses_csv

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### Limitation

-- COMMAND ----------

DESCRIBE EXTENDED courses_csv

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Impact of not having a Delta table

-- COMMAND ----------

-- MAGIC %python
-- MAGIC files = dbutils.fs.ls(f"{dataset_school}/courses-csv")
-- MAGIC display(files)

-- COMMAND ----------

SELECT COUNT(1) FROM courses_csv

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.cp(f"{dataset_school}/courses-csv/export_001.csv",
-- MAGIC               f"{dataset_school}/courses-csv/copy_001.csv")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC files = dbutils.fs.ls(f"{dataset_school}/courses-csv")
-- MAGIC display(files)

-- COMMAND ----------

SELECT COUNT(1) FROM courses_csv

-- COMMAND ----------

REFRESH TABLE courses_csv

-- COMMAND ----------

SELECT COUNT(1) FROM courses_csv

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Hybrid approach

-- COMMAND ----------

CREATE TEMP VIEW courses_tmp_vw
  (course_id STRING, title STRING, instructor STRING, category STRING, price DOUBLE)
USING CSV
OPTIONS (
 path = "${dataset.school}/courses-csv/export_*.csv",
 header = "true",
 delimiter = ";"
);

CREATE TABLE courses AS
 SELECT * FROM courses_tmp_vw;

-- COMMAND ----------

DESCRIBE EXTENDED courses

-- COMMAND ----------

SELECT * FROM courses
