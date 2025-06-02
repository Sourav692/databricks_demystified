-- Creating a new catalog
CREATE CATALOG IF NOT EXISTS hr_catalog;

-- Verifying the created catalog
SHOW CATALOGS;

-- Granting permissions
GRANT CREATE SCHEMA, CREATE TABLE, USE CATALOG ON CATALOG hr_catalog
TO `account users`;

SHOW GRANT ON CATALOG hr_catalog;

-- Creating schemas
CREATE SCHEMA IF NOT EXISTS hr_catalog.hr_db;

SHOW SCHEMAS IN hr_catalog;

-- Managing Delta Tables
CREATE TABLE IF NOT EXISTS hr_catalog.hr_db.jobs
(id INT, title STRING, min_salary DOUBLE, max_salary DOUBLE);

INSERT INTO hr_catalog.hr_db.jobs
VALUES (1, "Software Engineer", 3000, 5000),
       (2, "Data Engineer", 3500, 5500),
       (3, "Web Developer", 2800, 4800);

DESCRIBE EXTENDED hr_catalog.hr_db.jobs;

-- Dropping tables
DROP TABLE hr_catalog.hr_db.jobs;

UNDROP TABLE hr_catalog.hr_db.jobs;

SELECT * FROM hr_catalog.hr_db.jobs;


