# ingestion/ingest.py
# ------------------------------------------------------------------
# Layer 1: Ingest a CSV into HDFS and profile its columns, types,
#          nulls, and duplicate rows.
# ------------------------------------------------------------------

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from config.config import HDFS_RAW_DIR, HIVE_METASTORE_URI


def get_spark():
    """Create a Spark session wired to the Hive metastore."""
    return (
        SparkSession.builder
        .appName("TarkAI-Ingestion")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def get_table_name(local_path):
    """Turn a file path into a lowercase table name (file name without extension)."""
    return os.path.splitext(os.path.basename(local_path))[0].lower()


def detect_encoding(local_path):
    """Guess a CSV's text encoding so Spark can read non-UTF-8 files.

    Excel-exported CSVs are frequently Windows-1252 / Latin-1 (or UTF-16 for
    the "Unicode Text" export), not UTF-8. We read the raw bytes once and
    return the first encoding that decodes the whole file. latin-1 is last
    because it can decode any byte, so it is a safe final fallback.
    """
    if local_path.startswith("file://"):
        local_path = local_path[len("file://"):]
    with open(local_path, "rb") as f:
        raw = f.read()
    # UTF-16 files start with a BOM (or are full of NUL bytes) - spot them
    # first, otherwise cp1252 would "succeed" and produce NUL-filled columns.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:4096]:
        return "utf-16"
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(local_path, encoding):
    """Guess the column separator (',', ';', tab or '|') from a sample.

    European Excel exports frequently use ';' instead of ',', which would make
    Spark read each whole line as a single column. We sniff a small sample of
    the file and fall back to a comma when detection is not confident.
    """
    import csv
    if local_path.startswith("file://"):
        local_path = local_path[len("file://"):]
    with open(local_path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


# Python codec names (used above) -> the charset names Spark/Java understand.
# Spark rejects Python-only names like "utf-8-sig", so we always translate.
SPARK_ENCODINGS = {
    "utf-8-sig": "UTF-8",
    "utf-8": "UTF-8",
    "cp1252": "windows-1252",
    "latin-1": "ISO-8859-1",
    "utf-16": "UTF-16",
}


def read_local_csv(spark, local_path):
    """Read a CSV from the local file system (not HDFS) into a dataframe."""
    # Prefix with file:// so Spark reads from local FS, not HDFS
    local_uri = local_path if local_path.startswith("file://") else f"file://{os.path.abspath(local_path)}"
    # Detect encoding + delimiter so odd (Excel/Latin-1/semicolon) files load
    encoding = detect_encoding(local_path)
    delimiter = detect_delimiter(local_path, encoding)
    # Spark uses Java charset names, so translate the Python name we detected
    spark_encoding = SPARK_ENCODINGS.get(encoding, "UTF-8")
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("encoding", spark_encoding)
        .option("sep", delimiter)
        .csv(local_uri)
    )
    # A UTF-8 BOM can cling to the first column header - strip it if present
    if df.columns and df.columns[0].startswith("\ufeff"):
        df = df.withColumnRenamed(df.columns[0], df.columns[0].lstrip("\ufeff"))
    return df


def print_schema(df):
    """Print each column name with its detected datatype."""
    print("\nSchema (column -> datatype):")
    for field in df.schema.fields:
        print(f"  {field.name} -> {field.dataType.simpleString()}")


def print_null_counts(df):
    """Print how many null values each column has."""
    print("\nNull counts:")
    null_row = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in df.columns
    ]).collect()[0]
    for c in df.columns:
        print(f"  {c}: {null_row[c]}")


def print_duplicate_count(df):
    """Print how many fully duplicated rows are present."""
    dup_count = df.count() - df.dropDuplicates().count()
    print(f"\nDuplicate rows: {dup_count}")


def profile_dataframe(df, table_name):
    """Show a quick data profile: row/column counts, schema, nulls, duplicates."""
    print(f"\n=== Profiling: {table_name} ===")
    print(f"Rows: {df.count()} | Columns: {len(df.columns)}")
    print_schema(df)
    print_null_counts(df)
    print_duplicate_count(df)


def write_raw_to_hdfs(df, table_name):
    """Save an unchanged raw copy of the CSV into the HDFS raw directory."""
    raw_path = f"{HDFS_RAW_DIR}/{table_name}"
    df.write.mode("overwrite").option("header", True).csv(raw_path)
    print(f"\nRaw data written to HDFS: {raw_path}")


def ingest_csv(local_path):
    """Read a local CSV, profile it, and write the raw copy to HDFS."""
    spark = get_spark()

    table_name = get_table_name(local_path)
    df = read_local_csv(spark, local_path)

    profile_dataframe(df, table_name)
    write_raw_to_hdfs(df, table_name)

    spark.stop()
    return table_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit ingestion/ingest.py <local_csv_path>")
        sys.exit(1)
    ingest_csv(sys.argv[1])

