# ingestion/hive_loader.py
# ------------------------------------------------------------------
# Layer 3: Infer schema from cleaned data and create + load a Hive table
# Reads cleaned parquet from HDFS, creates the Hive table dynamically
# ------------------------------------------------------------------

import sys

from pyspark.sql import SparkSession
from config.config import HDFS_CLEANED_DIR, HIVE_DATABASE, HIVE_METASTORE_URI


def get_spark():
    """Create (or reuse) a Spark session that can talk to Hive."""
    return (
        SparkSession.builder
        .appName("TarkAI-HiveTableGen")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def spark_to_hive_type(spark_type):
    """Convert a Spark data type into the matching Hive data type (default STRING)."""
    type_map = {
        "int": "INT", "bigint": "BIGINT", "double": "DOUBLE",
        "float": "FLOAT", "boolean": "BOOLEAN", "date": "DATE",
        "timestamp": "TIMESTAMP",
    }
    return type_map.get(spark_type.simpleString(), "STRING")


def get_hdfs_handles(spark, path_str):
    """Return the Hadoop (FileSystem, Path) objects for an HDFS path string."""
    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jvm.java.net.URI(path_str), hadoop_conf
    )
    path = spark._jvm.org.apache.hadoop.fs.Path(path_str)
    return fs, path


def read_cleaned_data(spark, table_name):
    """Read the cleaned parquet for a table, erroring clearly if it is missing."""
    cleaned_path = f"{HDFS_CLEANED_DIR}/{table_name}"
    # Check the path exists before reading (avoids stale file-status cache issues)
    fs, hdfs_path = get_hdfs_handles(spark, cleaned_path)
    if not fs.exists(hdfs_path):
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. "
            f"Please re-run Layer 2 (cleaning) for '{table_name}' first."
        )
    return spark.read.parquet(cleaned_path)


def build_column_definitions(df):
    """Build the 'col_name HIVE_TYPE, ...' string used in the CREATE TABLE query."""
    return ", ".join(
        f"{f.name} {spark_to_hive_type(f.dataType)}" for f in df.schema.fields
    )


def remove_stale_table(spark, table_name):
    """Drop the table and delete any leftover warehouse folder from a past run."""
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    warehouse_dir = spark.conf.get("spark.sql.warehouse.dir")
    table_loc = f"{warehouse_dir}/{HIVE_DATABASE}.db/{table_name}"
    fs, stale_path = get_hdfs_handles(spark, table_loc)
    if fs.exists(stale_path):
        fs.delete(stale_path, True)
        print(f"  Removed stale location: {table_loc}")


def create_hive_table(table_name):
    """Full Layer 3 flow: read cleaned data -> create Hive table -> load it."""
    spark = get_spark()

    df = read_cleaned_data(spark, table_name)

    # make sure the target database exists and is selected
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DATABASE}")
    spark.sql(f"USE {HIVE_DATABASE}")

    # build the column list dynamically from the dataframe schema
    cols = build_column_definitions(df)
    print(f"\n=== Creating Hive table: {HIVE_DATABASE}.{table_name} ===")
    print(f"Columns: {cols}")

    # drop old table + clean up any stale files, then create fresh as Parquet
    remove_stale_table(spark, table_name)
    spark.sql(f"CREATE TABLE {table_name} ({cols}) STORED AS PARQUET")

    # load the cleaned data into the new table
    df.write.mode("overwrite").insertInto(f"{HIVE_DATABASE}.{table_name}")

    print(f"Table created and loaded. Row count: {spark.table(table_name).count()}")
    spark.sql(f"DESCRIBE {table_name}").show(truncate=False)

    spark.stop()
    return table_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit ingestion/hive_loader.py <table_name>")
        sys.exit(1)
    create_hive_table(sys.argv[1])
