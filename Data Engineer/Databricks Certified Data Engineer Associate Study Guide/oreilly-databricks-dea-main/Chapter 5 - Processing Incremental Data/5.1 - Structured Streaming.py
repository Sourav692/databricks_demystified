# Databricks notebook source
# MAGIC %md
# MAGIC # Implementing Structured Streaming

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <div  style="text-align: center;">
# MAGIC   <img src="https://raw.githubusercontent.com/derar-alhussein/oreilly-databricks-dea/main/Includes/Images/school_schema.png" alt="School Schema">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ../Includes/School-Setup

# COMMAND ----------

stream_df = spark.readStream.table("courses")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Data Manipulations in SQL

# COMMAND ----------

stream_df.createOrReplaceTempView("courses_streaming_tmp_vw")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM courses_streaming_tmp_vw

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying transformations

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT instructor, count(course_id) AS total_courses
# MAGIC FROM courses_streaming_tmp_vw
# MAGIC GROUP BY instructor

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT *
# MAGIC -- FROM courses_streaming_tmp_vw
# MAGIC -- ORDER BY instructor

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Persisting streaming data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW instructor_counts_tmp_vw AS (
# MAGIC  SELECT instructor, count(course_id) AS total_courses
# MAGIC  FROM courses_streaming_tmp_vw
# MAGIC  GROUP BY instructor
# MAGIC )

# COMMAND ----------

result_stream_df = spark.table("instructor_counts_tmp_vw")

# COMMAND ----------

(result_stream_df.writeStream 
                 .trigger(processingTime='3 seconds')
                 .outputMode("complete")
                 .option("checkpointLocation", "dbfs:/mnt/DEA-Book/checkpoints/instructor_counts")
                 .table("instructor_counts")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM instructor_counts

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO courses
# MAGIC values ("C16", "Generative AI", "Pierre B.", "Computer Science", 25),
# MAGIC        ("C17", "Embedded Systems", "Julia S.", "Computer Science", 30),
# MAGIC        ("C18", "Virtual Reality", "Bernard M.", "Computer Science", 35)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM instructor_counts

# COMMAND ----------

# MAGIC %md
# MAGIC #### Incremental batch processing

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO courses
# MAGIC values ("C19", "Compiler Design", "Sophie B.", "Computer Science", 25),
# MAGIC        ("C20", "Signal Processing", "Sam M.", "Computer Science", 30),
# MAGIC        ("C21", "Operating Systems", "Mark H.", "Computer Science", 35)

# COMMAND ----------

(result_stream_df.writeStream                                    
                 .trigger(availableNow=True)
                 .outputMode("complete")
                 .option("checkpointLocation", "dbfs:/mnt/DEA-Book/checkpoints/instructor_counts")
                 .table("instructor_counts")
                 .awaitTermination()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(1) FROM instructor_counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Data Manipulations in Python

# COMMAND ----------

import pyspark.sql.functions as F

output_stream_df = (stream_df.groupBy("instructor")
                   .agg(F.count("course_id").alias("total_courses")))

# COMMAND ----------

display(output_stream_df)

# COMMAND ----------

(output_stream_df.writeStream                                    
                 .trigger(availableNow=True)
                 .outputMode("complete")
                 .option("checkpointLocation", "dbfs:/mnt/DEA-Book/checkpoints/instructor_counts_py")
                 .table("instructor_counts_py")
                 .awaitTermination()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM instructor_counts_py

# COMMAND ----------

instructor_counts_df = spark.read.table("instructor_counts_py")

display(instructor_counts_df)
