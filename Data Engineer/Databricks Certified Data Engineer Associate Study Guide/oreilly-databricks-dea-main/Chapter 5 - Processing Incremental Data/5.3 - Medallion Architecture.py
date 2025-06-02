# Databricks notebook source
# MAGIC %md
# MAGIC # Building Medallion Architectures

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <div  style="text-align: center;">
# MAGIC   <img src="https://raw.githubusercontent.com/derar-alhussein/oreilly-databricks-dea/main/Includes/Images/school_schema.png" alt="School Schema">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ../Includes/School-Setup

# COMMAND ----------

files = dbutils.fs.ls(f"{dataset_school}/enrollments-json-raw")
display(files)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Establishing the bronze layer

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Configuring Auto Loader

# COMMAND ----------

import pyspark.sql.functions as F

(spark.readStream
           .format("cloudFiles")
           .option("cloudFiles.format", "json")
           .option("cloudFiles.inferColumnTypes","true")
           .option("cloudFiles.schemaLocation", f"{checkpoint_path}/enrollments_bronze")
           .load(f"{dataset_school}/enrollments-json-raw")
           .select("*",
                   F.current_timestamp().alias("arrival_time"),
                   F.input_file_name().alias("source_file"))
     .writeStream
           .format("delta")
           .option("checkpointLocation", f"{checkpoint_path}/enrollments_bronze")
           .outputMode("append")
           .table("enrollments_bronze")
)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM enrollments_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(1) FROM enrollments_bronze

# COMMAND ----------

load_new_data()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(1) FROM enrollments_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Creating a static lookup table

# COMMAND ----------

students_lookup_df = (spark.read
                       .format("json")
                       .load(f"{dataset_school}/students-json"))

# COMMAND ----------

display(students_lookup_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transitioning to the silver layer

# COMMAND ----------

enrollments_enriched_df = (spark.readStream
     .table("enrollments_bronze")
     .where("quantity > 0")
     .withColumn("formatted_timestamp", F.from_unixtime("enroll_timestamp", "yyyy-MM-dd HH:mm:ss").cast("timestamp") )
     .join(students_lookup_df, "student_id")
     .select("enroll_id", "quantity", "student_id", "email", "formatted_timestamp", "courses")
)

# COMMAND ----------

(enrollments_enriched_df.writeStream
                       .format("delta")
                       .option("checkpointLocation", f"{checkpoint_path}/enrollments_silver")
                       .outputMode("append")
                       .table("enrollments_silver"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM enrollments_silver

# COMMAND ----------

load_new_data()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(1) FROM enrollments_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advancing to the gold layer

# COMMAND ----------

enrollments_agg_df = (spark.readStream
                            .table("enrollments_silver")
                            .withColumn("day", F.date_trunc("DD", "formatted_timestamp"))
                            .groupBy("student_id", "email", "day")
                            .agg(F.sum("quantity").alias("courses_counts"))
                            .select("student_id", "email", "day", "courses_counts")
)

# COMMAND ----------

(enrollments_agg_df.writeStream
                   .format("delta")
                   .outputMode("complete")
                   .option("checkpointLocation", f"{checkpoint_path}/daily_student_courses")
                   .trigger(availableNow=True)
                   .table("daily_student_courses"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM daily_student_courses

# COMMAND ----------

load_new_data()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM daily_student_courses

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Stopping active streams

# COMMAND ----------

for s in spark.streams.active:
   print("Stopping stream: " + s.id)
   s.stop()
   s.awaitTermination()
