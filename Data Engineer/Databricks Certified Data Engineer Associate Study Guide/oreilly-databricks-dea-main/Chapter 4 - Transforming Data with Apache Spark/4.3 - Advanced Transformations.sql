-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Performing Advanced ETL Transformations

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div  style="text-align: center;">
-- MAGIC   <img src="https://raw.githubusercontent.com/derar-alhussein/oreilly-databricks-dea/main/Includes/Images/school_schema.png" alt="School Schema">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %run ../Includes/School-Setup

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Dealing with Nested JSON Data

-- COMMAND ----------

SELECT * FROM students

-- COMMAND ----------

DESCRIBE students

-- COMMAND ----------

SELECT student_id, profile:first_name, profile:address:country
FROM students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Parsing JSON into Struct Type

-- COMMAND ----------

SELECT profile 
FROM students 
LIMIT 1

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW parsed_students AS
 SELECT student_id, from_json(profile, schema_of_json('{"first_name":"Sarah",
 "last_name":"Lundi", "gender":"Female", "address":{"street":"8 Greenbank Road",
 "city":"Ottawa", "country":"Canada"}}')) AS profile_struct
 FROM students;

SELECT * FROM parsed_students

-- COMMAND ----------

DESCRIBE parsed_students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Interacting with Struct Types

-- COMMAND ----------

SELECT student_id, profile_struct.first_name, profile_struct.address.country
FROM parsed_students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Flattening Struct Types

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW students_final AS
 SELECT student_id, profile_struct.*
 FROM parsed_students;

SELECT * FROM students_final

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Leveraging the explode Function

-- COMMAND ----------

SELECT enroll_id, student_id, courses
FROM enrollments

-- COMMAND ----------

SELECT enroll_id, student_id, explode(courses) AS course
FROM enrollments

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Aggregating Unique Values 

-- COMMAND ----------

SELECT student_id,
 collect_set(enroll_id) AS enrollments_set,
 collect_set(courses.course_id) AS courses_set
FROM enrollments
GROUP BY student_id

-- COMMAND ----------

SELECT student_id,
 collect_set(courses.course_id) As before_flatten,
 array_distinct(flatten(collect_set(courses.course_id))) AS after_flatten
FROM enrollments
GROUP BY student_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Mastering Join Operations in Spark SQL

-- COMMAND ----------

CREATE OR REPLACE VIEW enrollments_enriched AS
SELECT *
FROM (
  SELECT *, explode(courses) AS course
  FROM enrollments) e
INNER JOIN courses c
ON e.course.course_id = c.course_id;

SELECT * FROM enrollments_enriched

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exploring Set Operations in Spark SQL

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW enrollments_updates
AS SELECT * FROM parquet.`${dataset.school}/enrollments-new`;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Union operation

-- COMMAND ----------

SELECT * FROM enrollments
UNION ALL
SELECT * FROM enrollments_updates

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Intersect operation

-- COMMAND ----------

SELECT * FROM enrollments
INTERSECT
SELECT * FROM enrollments_updates

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Minus operation

-- COMMAND ----------

SELECT * FROM enrollments
MINUS
SELECT * FROM enrollments_updates

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Changing Data Perspectives

-- COMMAND ----------

SELECT * FROM (
 SELECT student_id, course.course_id AS course_id, course.subtotal AS subtotal
 FROM enrollments_enriched
)
PIVOT (
 sum(subtotal) FOR course_id IN (
   'C01', 'C02', 'C03', 'C04', 'C05', 'C06',
   'C07', 'C08', 'C09', 'C10', 'C11', 'C12')
)

