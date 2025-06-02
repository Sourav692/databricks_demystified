-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Exploring Views

-- COMMAND ----------

USE CATALOG hive_metastore;

CREATE TABLE IF NOT EXISTS cars
(id INT, model STRING, brand STRING, year INT);

INSERT INTO cars
VALUES (1, 'Cybertruck', 'Tesla', 2024),
     (2, 'Model S', 'Tesla', 2023),
     (3, 'Model Y', 'Tesla', 2022),
     (4, 'Model X 75D', 'Tesla', 2017),
     (5, 'G-Class G63', 'Mercedes-Benz', 2024),
     (6, 'E-Class E200', 'Mercedes-Benz', 2023),
     (7, 'C-Class C300', 'Mercedes-Benz', 2016),
     (8, 'Everest', 'Ford', 2023),
     (9, 'Puma', 'Ford', 2021),
     (10, 'Focus', 'Ford', 2019)

-- COMMAND ----------

SHOW TABLES

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## View Types

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Stored views

-- COMMAND ----------

CREATE VIEW view_tesla_cars
AS SELECT *
   FROM cars
   WHERE brand = 'Tesla';

-- COMMAND ----------

SHOW TABLES;

-- COMMAND ----------

SELECT * FROM view_tesla_cars;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Temporary views

-- COMMAND ----------

CREATE TEMP VIEW temp_view_cars_brands
AS  SELECT DISTINCT brand
   FROM cars;

SELECT * FROM temp_view_cars_brands;

-- COMMAND ----------

SHOW TABLES;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Global temporary views

-- COMMAND ----------

CREATE GLOBAL TEMP VIEW global_temp_view_recent_cars
AS SELECT * FROM cars
   WHERE year >= 2022
   ORDER BY year DESC;

-- COMMAND ----------

SELECT * FROM global_temp.global_temp_view_recent_cars;

-- COMMAND ----------

SHOW TABLES;

-- COMMAND ----------

SHOW TABLES IN global_temp;
