-- Creating data objects
CREATE DATABASE IF NOT EXISTS hive_metastore.hr_db
LOCATION 'dbfs:/mnt/demo/hr_db.db';

CREATE TABLE hive_metastore.hr_db.employees
(id INT, name STRING, salary DOUBLE, city STRING);

INSERT INTO hive_metastore.hr_db.employees
VALUES (1, "Felipe", 3000, "London"),
      (2, "Sachin", 3400, "New York"),
      (3, "Anna", 3600, "London"),
      (4, "Hong-Thai", 3200, "London"),
      (5, "Charlotte", 3500, "New York"),
      (6, "Amine", 3400, "New York"),
      (7, "Emily", 3200, "London");

CREATE VIEW hive_metastore.hr_db.london_employees_vw
AS SELECT * FROM hive_metastore.hr_db.employees WHERE city = 'London';

-- Configuring object permissions
---- Granting privileges to a group:
GRANT SELECT, MODIFY, READ_METADATA, CREATE
ON SCHEMA hive_metastore.hr_db TO hr_team;

GRANT USAGE ON SCHEMA hive_metastore.hr_db TO hr_team;

---- Granting privileges to an individual user
GRANT SELECT
ON VIEW hive_metastore.hr_db.london_employees_vw TO `eve@example.com`;

---- Reviewing assigned permissions
SHOW GRANTS ON SCHEMA hive_metastore.hr_db;

SHOW GRANTS ON VIEW hive_metastore.hr_db.london_employees_vw;


