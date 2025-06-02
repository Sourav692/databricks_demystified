-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Developing SQL UDFs

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
-- MAGIC ## Creating UDFs

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gpa_to_percentage(gpa DOUBLE)
RETURNS INT

RETURN cast(round(gpa * 25) AS INT)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Applying UDFs

-- COMMAND ----------

SELECT student_id, gpa, gpa_to_percentage(gpa) AS percentage_score
FROM students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Understanding UDFs

-- COMMAND ----------

DESCRIBE FUNCTION gpa_to_percentage

-- COMMAND ----------

DESCRIBE FUNCTION EXTENDED gpa_to_percentage

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Complex Logic UDFs

-- COMMAND ----------

CREATE OR REPLACE FUNCTION get_letter_grade(gpa DOUBLE)
RETURNS STRING
RETURN CASE
        WHEN gpa >= 3.5 THEN "A"
        WHEN gpa >= 2.75 AND gpa < 3.5 THEN "B"
        WHEN gpa >= 2 AND gpa < 2.75 THEN "C"
        ELSE "F"
     END

-- COMMAND ----------

SELECT student_id, gpa, get_letter_grade(gpa) AS letter_grade
FROM students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Dropping UDFs

-- COMMAND ----------

DROP FUNCTION gpa_to_percentage;
DROP FUNCTION get_letter_grade;
