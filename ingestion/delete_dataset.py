# ingestion/delete_dataset.py
# -----------------------------------------------------------------------------
# DATASET REMOVAL (housekeeping)
#
# When we ingest many CSVs the workspace gets crowded and confusing. This file
# fully deletes ONE dataset - undoing everything the pipeline created for it:
#   1. the Hive table (Layer 3) and its warehouse folder
#   2. the HDFS raw copy (Layer 1) and the cleaned copy (Layer 2)
#   3. the local uploaded CSV and its data-quality report
#   4. its MySQL metadata rows (schema, business terms, KPIs, relationships)
#
# delete_dataset(table_name) is the one function the Streamlit app calls.
# We reuse the SAME running Spark session as the rest of the app, so we must
# NOT stop it here - other pages still need it.
# -----------------------------------------------------------------------------

import os
import glob
from pathlib import Path

from dotenv import load_dotenv

# Load the Spark/Hadoop/Hive paths from the project-root .env before Spark
# starts (same approach the executor uses), so this also works when the file
# is run on its own from the command line.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Safety net: --packages makes Spark try to download jars from the internet,
# which crashes when the machine is offline. Force the plain shell instead.
if "--packages" in os.environ.get("PYSPARK_SUBMIT_ARGS", ""):
    os.environ["PYSPARK_SUBMIT_ARGS"] = "pyspark-shell"

from config.config import (
    HDFS_RAW_DIR, HDFS_CLEANED_DIR, HIVE_DATABASE, HIVE_METASTORE_URI, DATA_DIR,
)
from config.db import execute, fetch_all


def get_spark():
    """Return the shared Hive-enabled Spark session (reused, never stopped here)."""
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName("TarkAI-DeleteDataset")
        .master("local[*]")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def list_datasets():
    """Return the names of all datasets currently registered (from schema_context)."""
    rows = fetch_all("SELECT DISTINCT table_name FROM schema_context ORDER BY table_name")
    return [r[0] for r in rows]


def _delete_hdfs_path(spark, path_str):
    """Best-effort delete of one HDFS directory. Returns True if it removed something."""
    try:
        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(path_str), hadoop_conf
        )
        path = spark._jvm.org.apache.hadoop.fs.Path(path_str)
        if fs.exists(path):
            fs.delete(path, True)
            print(f"  Removed HDFS path: {path_str}")
            return True
    except Exception as e:
        print(f"  Could not delete HDFS path {path_str}: {e}")
    return False


def drop_hive_table(spark, table_name):
    """Drop the Hive table and delete its leftover warehouse folder if any."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DATABASE}")
    spark.sql(f"USE {HIVE_DATABASE}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    warehouse_dir = spark.conf.get("spark.sql.warehouse.dir")
    table_loc = f"{warehouse_dir}/{HIVE_DATABASE}.db/{table_name}"
    _delete_hdfs_path(spark, table_loc)
    print(f"  Dropped Hive table: {HIVE_DATABASE}.{table_name}")


def delete_hdfs_data(spark, table_name):
    """Delete the raw (Layer 1) and cleaned (Layer 2) HDFS copies of the dataset."""
    _delete_hdfs_path(spark, f"{HDFS_RAW_DIR}/{table_name}")
    _delete_hdfs_path(spark, f"{HDFS_CLEANED_DIR}/{table_name}")


def delete_local_files(table_name):
    """Delete the quality report and the original uploaded CSV for this table.

    The CSV keeps its original file name, so we match any *.csv in the data
    folder whose name (without extension, lowercased) equals the table name.
    Returns the list of file paths we removed.
    """
    removed = []

    report_path = os.path.join(DATA_DIR, f"{table_name}_quality_report.json")
    if os.path.exists(report_path):
        os.remove(report_path)
        removed.append(report_path)

    for csv_path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
        if stem == table_name:
            os.remove(csv_path)
            removed.append(csv_path)

    return removed


def delete_metadata(table_name):
    """Remove this table's rows from every MySQL metadata table."""
    execute("DELETE FROM schema_context WHERE table_name=%s", (table_name,))
    execute("DELETE FROM business_context WHERE table_name=%s", (table_name,))
    execute("DELETE FROM kpi_context WHERE table_name=%s", (table_name,))
    execute(
        "DELETE FROM relationship_context WHERE parent_table=%s OR child_table=%s",
        (table_name, table_name),
    )
    print("  Removed MySQL metadata (schema, business, KPI, relationships)")


def delete_dataset(table_name):
    """Delete a dataset everywhere the pipeline stored it and report what was removed.

    Runs the cleanup in order: Hive table (+ warehouse) -> HDFS raw & cleaned ->
    local CSV & quality report -> MySQL metadata rows. The shared Spark session
    is left running because the rest of the app keeps using it.

    Returns a small summary dict the UI can show.
    """
    print(f"\n=== Deleting dataset: {table_name} ===")
    spark = get_spark()

    drop_hive_table(spark, table_name)
    delete_hdfs_data(spark, table_name)
    spark.catalog.clearCache()  # forget any cached file listings, just to be safe

    local_removed = delete_local_files(table_name)
    delete_metadata(table_name)

    print(f"=== Dataset '{table_name}' deleted ===")
    return {
        "table": table_name,
        "hive_table_dropped": True,
        "local_files_removed": local_removed,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.delete_dataset <table_name>")
        sys.exit(1)
    delete_dataset(sys.argv[1])
