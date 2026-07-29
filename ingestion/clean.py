# ingestion/clean.py
# ------------------------------------------------------------------
# Layer 2: Clean a raw dataset using PySpark
#
# What this script does (in order):
#   1. Remove duplicate rows
#   2. Trim text columns, fill blanks/nulls with 'unknown'
#   3. Fill numeric nulls with 0
#   4. Rename columns to snake_case
#   5. Detect and cap outliers using the IQR method
#   6. Save a small data quality report as JSON
# ------------------------------------------------------------------

import os
import sys
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, lit
from pyspark.sql.types import StringType, NumericType
from config.config import HDFS_RAW_DIR, HDFS_CLEANED_DIR, HIVE_METASTORE_URI, DATA_DIR


def get_spark():
    """Create (or reuse) a Spark session that can talk to Hive."""
    return (
        SparkSession.builder
        .appName("TarkAI-Cleaning")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def get_string_columns(df):
    """Return the names of all text (string) columns."""
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]


def get_numeric_columns(df):
    """Return the names of all numeric columns."""
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]


def remove_duplicates(df, before, report):
    """Step 1: Drop exact duplicate rows."""
    df = df.dropDuplicates()
    removed = before - df.count()
    report["steps"].append({"step": "Remove Duplicates", "removed": removed})
    print(f"  Step 1: Removed {removed} duplicate rows")
    return df


def clean_string_columns(df, report):
    """Step 2: Trim spaces in text columns and replace blanks/nulls with 'unknown'."""
    string_cols = get_string_columns(df)
    for name in string_cols:
        df = df.withColumn(name, trim(col(name)))
        df = df.withColumn(
            name,
            when(col(name).isNull() | (col(name) == ""), "unknown").otherwise(col(name)),
        )
    report["steps"].append({"step": "Trim & Fill Strings", "columns_affected": len(string_cols)})
    print(f"  Step 2: Trimmed {len(string_cols)} string columns")
    return df


def fill_numeric_nulls(df, report):
    """Step 3: Replace nulls in numeric columns with 0."""
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        df = df.fillna(0, subset=numeric_cols)
        report["steps"].append({
            "step": "Fill Numeric Nulls",
            "columns_affected": len(numeric_cols),
            "fill_value": 0,
        })
        print(f"  Step 3: Filled nulls in {len(numeric_cols)} numeric columns")
    return df


def normalize_column_names(df, report):
    """Step 4: Convert column names to clean snake_case."""
    renamed = 0
    for c in df.columns:
        new_c = c.strip().lower().replace(" ", "_").replace("-", "_")
        new_c = "".join(ch for ch in new_c if ch.isalnum() or ch == "_")
        if new_c != c:
            df = df.withColumnRenamed(c, new_c)
            renamed += 1
    report["steps"].append({"step": "Normalize Column Names", "columns_renamed": renamed})
    print(f"  Step 4: Renamed {renamed} columns to snake_case")
    return df


def cap_outliers(df, report):
    """Step 5: Find outliers with the IQR rule and cap them to the nearest bound."""
    # IQR method (suggested by our guide) works well for skewed data
    numeric_cols = get_numeric_columns(df)
    outlier_summary = {}
    for c in numeric_cols:
        quantiles = df.approxQuantile(c, [0.25, 0.75], 0.05)
        if len(quantiles) == 2 and quantiles[0] is not None and quantiles[1] is not None:
            q1, q3 = quantiles
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = df.filter((col(c) < lower_bound) | (col(c) > upper_bound)).count()
                if outlier_count > 0:
                    df = df.withColumn(
                        c,
                        when(col(c) < lower_bound, lit(lower_bound))
                        .when(col(c) > upper_bound, lit(upper_bound))
                        .otherwise(col(c)),
                    )
                    outlier_summary[c] = {
                        "outliers_capped": outlier_count,
                        "lower_bound": round(lower_bound, 2),
                        "upper_bound": round(upper_bound, 2),
                    }
                    print(f"  Step 5: Capped {outlier_count} outliers in '{c}' "
                          f"[{round(lower_bound, 2)}, {round(upper_bound, 2)}]")
    report["steps"].append({
        "step": "Outlier Detection (IQR)",
        "columns_checked": len(numeric_cols),
        "outliers": outlier_summary,
    })
    return df


def save_cleaned_data(df, table_name):
    """Write the cleaned dataframe to HDFS as parquet (overwrite existing)."""
    cleaned_path = f"{HDFS_CLEANED_DIR}/{table_name}"
    df.write.mode("overwrite").parquet(cleaned_path)
    print(f"  Cleaned data saved to: {cleaned_path}")


def save_quality_report(report, table_name):
    """Save the data quality report as a JSON file in the local data folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    report_path = os.path.join(DATA_DIR, f"{table_name}_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Quality report saved to: {report_path}")


def clean_dataset(table_name):
    """Run the full cleaning pipeline for one table: read -> clean -> save."""
    spark = get_spark()

    raw_path = f"{HDFS_RAW_DIR}/{table_name}"
    df = spark.read.option("header", True).option("inferSchema", True).csv(raw_path)

    print(f"\n=== Cleaning: {table_name} ===")
    before = df.count()
    report = {
        "table": table_name,
        "rows_before": before,
        "columns": len(df.columns),
        "steps": [],
    }

    # run each cleaning step one by one
    df = remove_duplicates(df, before, report)
    df = clean_string_columns(df, report)
    df = fill_numeric_nulls(df, report)
    df = normalize_column_names(df, report)
    df = cap_outliers(df, report)

    after = df.count()
    report["rows_after"] = after
    print(f"\n  Rows before: {before} | after: {after} | removed: {before - after}")

    save_cleaned_data(df, table_name)
    save_quality_report(report, table_name)

    spark.stop()
    return table_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit ingestion/clean.py <table_name>")
        sys.exit(1)
    clean_dataset(sys.argv[1])
