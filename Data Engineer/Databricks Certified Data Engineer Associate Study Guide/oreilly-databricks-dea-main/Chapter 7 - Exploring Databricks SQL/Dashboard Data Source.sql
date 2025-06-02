SELECT "United Kingdom" region, student_id, f_name, l_name, order_date, courses_counts
FROM hive_metastore.school_dlt_db.uk_daily_student_courses

UNION

SELECT "France" region, student_id, f_name, l_name, order_date, courses_counts
FROM hive_metastore.school_dlt_db.fr_daily_student_courses