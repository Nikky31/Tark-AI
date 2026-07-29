# query_engine/executor.py
# ------------------------------------------------------------------
# Layer 9: Execute generated Hive SQL, time it, return results, and
#          log the query to MySQL for an audit trail.
# ------------------------------------------------------------------

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# LOAD ENVIRONMENT CONFIG (must happen before Spark starts)
# ---------------------------------------------------------------------
# All the Spark/Hadoop/Hive install paths live in the project-root .env
# file, NOT in this code. We load them here so they are set as real
# environment variables before PySpark reads them (PySpark grabs
# SPARK_HOME, HADOOP_HOME, PYSPARK_SUBMIT_ARGS, etc. the moment the JVM
# starts, so this has to run first).
#
# load_dotenv does not overwrite variables that are already set in the
# shell, so a real shell export still wins over the .env file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Safety net: --packages tells PySpark to download jars from Maven, which
# crashes the JVM when the machine is offline. If some .bashrc set it, we
# force it back to the plain shell so in-process Spark still works.
if "--packages" in os.environ.get("PYSPARK_SUBMIT_ARGS", ""):
    os.environ["PYSPARK_SUBMIT_ARGS"] = "pyspark-shell"

from config.config import (
    EXECUTION_ENGINE, HIVE_METASTORE_URI, HIVE_SERVER_HOST, HIVE_SERVER_PORT,
    HIVE_DATABASE,
)
from config.db import execute as db_execute


# ---------------------------------------------------------------------
# SPARK SESSION (created once, then reused)
# ---------------------------------------------------------------------

_spark = None


def _get_spark():
    """Return a live Spark session, recreating it if the old one went stale.

    We keep one session alive for speed. But a long-running session can die
    in the background, so we first poke the existing one and rebuild it if
    that check fails.
    """
    global _spark

    # if we already have a session, make sure it is still alive
    if _spark is not None:
        try:
            _spark.sparkContext._jsc.sc().isStopped()
        except Exception:
            print("[Spark] Detected stale session, recreating...")
            try:
                _spark.stop()
            except Exception:
                pass
            _spark = None

    # create a fresh session if we don't have one
    if _spark is None:
        from pyspark.sql import SparkSession
        SparkSession._instantiatedSession = None
        _spark = (
            SparkSession.builder
            .appName("TarkAI-Executor")
            .master("local[*]")
            .config("hive.metastore.uris", HIVE_METASTORE_URI)
            .config("spark.sql.catalogImplementation", "hive")
            .enableHiveSupport()
            .getOrCreate()
        )
    return _spark


# ---------------------------------------------------------------------
# STALE-FILE HANDLING
# ---------------------------------------------------------------------

def _is_stale_file_error(error):
    """Return True if the error is Spark complaining about a missing file.

    This happens when a table's parquet files were REWRITTEN (for example the
    dataset was re-uploaded, or the ingestion pipeline was re-run) while our
    long-lived Spark session still remembers the OLD file names. Spark's own
    message says to run 'REFRESH TABLE', so we detect that situation here and
    fix it automatically instead of failing.
    """
    message = str(error).lower()
    return (
        "does not exist" in message
        or "filenotfound" in message
        or "refresh table" in message
    )


def _refresh_all_tables(spark):
    """Clear Spark's cached file listing for every table in our database.

    After this, the next query re-reads the CURRENT parquet files instead of
    the stale ones, so a freshly re-uploaded dataset no longer breaks queries
    in the running app.
    """
    try:
        tables = spark.catalog.listTables(HIVE_DATABASE)
    except Exception as list_error:
        # If we cannot even list the tables, clear the whole cache instead.
        print(f"[Spark] Could not list tables ({list_error}); clearing all cache")
        spark.catalog.clearCache()
        return

    for table in tables:
        try:
            spark.catalog.refreshTable(f"{HIVE_DATABASE}.{table.name}")
        except Exception as refresh_error:
            print(f"[Spark] Could not refresh {table.name}: {refresh_error}")
    print(f"[Spark] Refreshed {len(tables)} table(s) after a stale-file error")


# ---------------------------------------------------------------------
# THE TWO WAYS WE CAN RUN SQL
# ---------------------------------------------------------------------

def _run_spark(sql):
    """Run SQL through the in-process Spark session and return a pandas DataFrame.

    If the run fails because the table's files changed since Spark cached
    them, we refresh the tables once and retry. Any other error is re-raised
    so the real problem is reported to the user.
    """
    spark = _get_spark()
    try:
        return spark.sql(sql).toPandas()
    except Exception as error:
        if _is_stale_file_error(error):
            print("[Spark] Stale file cache detected, refreshing and retrying...")
            _refresh_all_tables(spark)
            return spark.sql(sql).toPandas()
        raise


def _run_pyhive(sql):
    """Run SQL through a direct PyHive connection and return a pandas DataFrame."""
    from pyhive import hive
    import pandas as pd

    conn = hive.Connection(host=HIVE_SERVER_HOST, port=HIVE_SERVER_PORT)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------
# AUDIT LOG + PUBLIC ENTRY POINT
# ---------------------------------------------------------------------

def log_query(user_query, sql, engine, intent, exec_time, row_count, status, error=""):
    """Persist an audit log row in MySQL."""
    db_execute(
        "INSERT INTO query_logs "
        "(user_query, generated_sql, engine, intent, execution_time_sec, "
        " row_count, status, error_message) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (user_query, sql, engine, intent, exec_time, row_count, status, error),
    )


def execute_query(sql, user_query="", engine="rule", intent=""):
    """Execute SQL on Hive (via Spark or PyHive), return a result dict, and log it.

    The engine is chosen by EXECUTION_ENGINE in config: "hive" uses PyHive,
    anything else uses Spark. We always time the run and write one audit row
    (SUCCESS or FAILED) to MySQL.

    Returns: {success, dataframe, row_count, execution_time, error}
    """
    start = time.time()
    result = {"success": False, "dataframe": None, "row_count": 0,
              "execution_time": 0.0, "error": ""}
    try:
        if EXECUTION_ENGINE == "hive":
            df = _run_pyhive(sql)
        else:
            df = _run_spark(sql)

        elapsed = round(time.time() - start, 4)
        result.update({
            "success": True, "dataframe": df,
            "row_count": len(df), "execution_time": elapsed,
        })
        log_query(user_query, sql, engine, intent, elapsed, len(df), "SUCCESS")
    except Exception as e:
        elapsed = round(time.time() - start, 4)
        result.update({"execution_time": elapsed, "error": str(e)})
        log_query(user_query, sql, engine, intent, elapsed, 0, "FAILED", str(e))
        print(f"[Execution error] {e}")
    return result


if __name__ == "__main__":
    test_sql = (f"SELECT customer_id, SUM(sales_amount) AS revenue "
                f"FROM {HIVE_DATABASE}.sales GROUP BY customer_id ORDER BY revenue DESC")
    r = execute_query(test_sql, user_query="top customers by revenue", intent="Ranking")
    print("Success:", r["success"], "| rows:", r["row_count"], "| time:", r["execution_time"])
    if r["dataframe"] is not None:
        print(r["dataframe"])
