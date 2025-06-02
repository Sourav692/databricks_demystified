-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Extending DLT Pipelines with New Notebooks

-- COMMAND ----------

SET school.dataset_path=dbfs:/mnt/DE-Associate-Book/datasets/school;

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE courses_bronze
COMMENT "The raw courses data, ingested from CDC feed"
AS SELECT * FROM cloud_files("${school.dataset_path}/courses-cdc", "json")

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE courses_silver;

-- COMMAND ----------

APPLY CHANGES INTO LIVE.courses_silver
 FROM STREAM(LIVE.courses_bronze)
 KEYS (course_id)
 APPLY AS DELETE WHEN row_status = "DELETE"
 SEQUENCE BY row_time
 COLUMNS * EXCEPT (row_status, row_time)

-- COMMAND ----------

CREATE OR REPLACE MATERIALIZED VIEW instructor_counts_stats
 COMMENT "Number of courses per instructor"
AS SELECT instructor, count(*) as courses_count,
         current_timestamp() updated_time
 FROM LIVE.courses_silver
 GROUP BY instructor

-- COMMAND ----------

CREATE TEMPORARY LIVE VIEW courses_sales
 AS SELECT b.title, o.quantity
   FROM (
     SELECT *, explode(courses) AS course
     FROM LIVE.enrollments_cleaned) o
   INNER JOIN LIVE.courses_silver b
   ON o.course.course_id = b.course_id;
