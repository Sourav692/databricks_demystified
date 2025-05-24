## Chapter 2 - Applying Data Transformations using Delta Live Tables

In this chapter, we're going to dive straight into how DLT makes ingesting data from a variety of input sources simple and straightforward, whether its files landing in cloud storage, or connecting to an external storage system like a relational database management system (RDBMS). In the hands-on exercise, we'll take a look at how we can efficiently and accurately apply changes from our input data sources to downstream datasets using the `APPLY INTO` command.

You can always create a new notebook from scratch, but it's recommended to download and import the accompanying notebook samples:

- `02-Publishing Datasets to Unity Catalog.py` - A sample notebook which demonstrates the types of permissions available for securable objects in Unity Catalog.
- `03a-Generate Postgres Table.py` - An optional notebook for writing mock taxi driver information to a Postgres database
- `03b-Generate Mock Taxi Trip Data.py` - A taxi trip data generator notebook that will create random dataset of mock taxi trips.
- `03c-Taxi Trips DLT Pipelien.py` - A notebook containing the DLT pipeline definition for the hands-on exercise.

### Technical requirements
To follow along in this chapter, you will need to have Databricks workspace permissions to create and start an all-purpose cluster so that you can execute all of the accompanying notebook cells. You will also need permissions to create and run a new DLT pipeline using a cluster policy. It's recommended to have Unity Catalog permissions to create and use Catalogs, Schemas, and Tables.

### Expected costs
This chapter will create and run several new notebooks and Delta Live Table pipelines using the `Core` product edition. As a result, the pipelines are estimated to consume around 10-15 Databricks Units (DBUs).

Please see the Databricks documentation for the latest pricing figures: [Pricing calculator](https://www.databricks.com/product/pricing/product-pricing/instance-types).
