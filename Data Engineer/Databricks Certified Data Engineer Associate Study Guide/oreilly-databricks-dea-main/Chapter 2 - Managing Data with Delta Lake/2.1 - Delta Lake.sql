-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Working with Delta Lake Tables

-- COMMAND ----------

USE CATALOG hive_metastore

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Creating Tables

-- COMMAND ----------

CREATE TABLE product_info (
  product_id INT,
  product_name STRING,
  category STRING,
  price DOUBLE,
  quantity INT
)
USING DELTA;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Inserting Data

-- COMMAND ----------

INSERT INTO product_info (product_id, product_name, category, price, quantity)
VALUES (1, 'Winter Jacket', 'Clothing', 79.95, 100);


INSERT INTO product_info (product_id, product_name, category, price, quantity)
VALUES
 (2, 'Microwave', 'Kitchen', 249.75, 30),
 (3, 'Board Game', 'Toys', 29.99, 75),
 (4, 'Smartcar', 'Electronics', 599.99, 50);

-- COMMAND ----------

SELECT * FROM product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exploring the Table Directory

-- COMMAND ----------

DESCRIBE DETAIL product_info

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Updating Delta Lake Tables

-- COMMAND ----------

UPDATE product_info
SET price = price + 10
WHERE product_id = 3

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'

-- COMMAND ----------

DESCRIBE DETAIL product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exploring Table History

-- COMMAND ----------

DESCRIBE HISTORY product_info

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info/_delta_log'

-- COMMAND ----------

-- MAGIC %fs head 'dbfs:/user/hive/warehouse/product_info/_delta_log/00000000000000000003.json'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Exploring Delta Time Travel

-- COMMAND ----------

DESCRIBE HISTORY product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Querying Older Versions

-- COMMAND ----------

SELECT * FROM product_info VERSION AS OF 2

-- COMMAND ----------

SELECT * FROM product_info@v2

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Rolling Back to Previous Versions

-- COMMAND ----------

DELETE FROM product_info

-- COMMAND ----------

DESCRIBE HISTORY product_info

-- COMMAND ----------

RESTORE TABLE product_info TO VERSION AS OF 3

-- COMMAND ----------

SELECT * FROM product_info

-- COMMAND ----------

DESCRIBE HISTORY product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Optimizing Delta Lake Tables

-- COMMAND ----------

DESCRIBE DETAIL product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Z-Order Indexing

-- COMMAND ----------

OPTIMIZE product_info
ZORDER BY product_id

-- COMMAND ----------

DESCRIBE DETAIL product_info

-- COMMAND ----------

DESCRIBE HISTORY product_info

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Vacuuming

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'

-- COMMAND ----------

VACUUM product_info

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'

-- COMMAND ----------

VACUUM product_info RETAIN 0 HOURS

-- COMMAND ----------

SET spark.databricks.delta.retentionDurationCheck.enabled = false

-- COMMAND ----------

VACUUM product_info RETAIN 0 HOURS

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'

-- COMMAND ----------

SELECT * FROM product_info@v1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Dropping Delta Lake Tables

-- COMMAND ----------

DROP TABLE product_info

-- COMMAND ----------

-- MAGIC %fs ls 'dbfs:/user/hive/warehouse/product_info'
