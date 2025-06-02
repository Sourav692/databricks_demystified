-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Implementing DLT Pipelines

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div  style="text-align: center;">
-- MAGIC   <img src="https://raw.githubusercontent.com/derar-alhussein/oreilly-databricks-dea/main/Includes/Images/school_schema.png" alt="School Schema">
-- MAGIC </div>

-- COMMAND ----------

SET school.dataset_path=dbfs:/mnt/DE-Associate-Book/datasets/school;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Bronze layer

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Creating a streaming table

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE enrollments_raw
COMMENT "The raw courses enrollments, ingested from enrollments-dlt-raw folder"
AS SELECT * FROM cloud_files("${school.dataset_path}/enrollments-dlt-raw",
                            "json",
                            map("cloudFiles.inferColumnTypes", "true"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Creating a materialized view

-- COMMAND ----------

CREATE OR REPLACE MATERIALIZED VIEW students
COMMENT "The students lookup table, ingested from students-json"
AS SELECT * FROM json.`${school.dataset_path}/students-json`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Silver layer

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE enrollments_cleaned (
 CONSTRAINT valid_order_number EXPECT (enroll_id IS NOT NULL) ON VIOLATION DROP ROW
)
COMMENT "The cleaned courses enrollments with valid enroll_id"
AS
 SELECT enroll_id, quantity, o.student_id, c.profile:first_name as f_name, c.profile:last_name as l_name,
        cast(from_unixtime(enroll_timestamp, 'yyyy-MM-dd HH:mm:ss') AS timestamp) formatted_timestamp,
        o.courses, c.profile:address:country as country
 FROM STREAM(LIVE.enrollments_raw) o
 LEFT JOIN LIVE.students c
   ON o.student_id = c.student_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC ## Gold layer

-- COMMAND ----------

CREATE OR REPLACE MATERIALIZED VIEW uk_daily_student_courses
COMMENT "Daily number of courses per student in United Kingdom"
AS
 SELECT student_id, f_name, l_name, date_trunc("DD", formatted_timestamp) order_date, sum(quantity) courses_counts
 FROM LIVE.enrollments_cleaned
 WHERE country = "United Kingdom"
 GROUP BY student_id, f_name, l_name, date_trunc("DD", formatted_timestamp)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Configuring DLT Pipelines

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Modifying DLT pipelines

-- COMMAND ----------

CREATE OR REPLACE MATERIALIZED VIEW fr_daily_student_courses
COMMENT "Daily number of courses per student in France"
AS
 SELECT student_id, f_name, l_name, date_trunc("DD", formatted_timestamp) order_date, sum(quantity) courses_counts
 -- FROM enrollments_cleaned
 FROM LIVE.enrollments_cleaned
 WHERE country = "France"
 GROUP BY student_id, f_name, l_name, date_trunc("DD", formatted_timestamp)
