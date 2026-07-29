# context/relationships.py
# =====================================================================
# LAYER 4 : RELATIONSHIP DISCOVERY
# ---------------------------------------------------------------------
# What this file does (in plain English):
#
# A database usually has several tables that are linked together, for
# example a "customer" table and a "fact_sales" table both have a
# "customer_id" column. Knowing these links lets us JOIN tables later in
# SQL generation. But nobody told us the links, so we DISCOVER them
# automatically:
#
#   1. Look at every table and pick out the columns that look like keys
#      (names ending in _id, id, code, key).
#   2. Take every PAIR of tables. If they share a key column with the
#      same name AND the same data type, they might be related.
#   3. Measure how much their actual values overlap. A high overlap means
#      it is almost certainly a real foreign-key relationship.
#         confidence = (shared distinct values / smaller set) x 100
#   4. Save each relationship we find into the MySQL relationship_context
#      table (parent_table, child_table, join_key, confidence).
#
# Which table is the PARENT?
#   In a star schema the big FACT table (fact_sales, one row per sale)
#   points to several small dimension / lookup tables. We treat the
#   BIGGER table (more rows) as the parent 'hub' and the smaller lookup
#   table as the child, so every link reads the same way, e.g.
#   fact_sales.customer_id vs customer.customer_id -> fact_sales is the
#   parent, customer is the child, join key = customer_id.
# =====================================================================

from itertools import combinations

from pyspark.sql import SparkSession
from config.config import HIVE_DATABASE, HIVE_METASTORE_URI
from config.db import execute


# A column is a possible join key if its name ends with (or equals) one of these.
KEY_HINTS = ("_id", "id", "code", "key")


def get_spark():
    """Start (or reuse) a Spark session pointed at our Hive metastore."""
    return (
        SparkSession.builder
        .appName("TarkAI-Relationships")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def is_key_like(column_name):
    """True if a column name looks like a join key (case-insensitive KEY_HINTS)."""
    name = column_name.lower()
    return any(name.endswith(hint) or name == hint for hint in KEY_HINTS)


def get_key_columns(spark, table_name):
    """Return the key-like columns of a table as (name, data_type) pairs.

    We keep the data type too, so we only compare columns whose types match.
    """
    return [
        (f.name, f.dataType.simpleString())
        for f in spark.table(table_name).schema.fields
        if is_key_like(f.name)
    ]


def get_row_count(spark, table_name):
    """Return the row count of a table (used to spot the bigger 'fact' table)."""
    return spark.sql(f"SELECT COUNT(*) FROM {table_name}").collect()[0][0]


def get_distinct_values(spark, table_name, column_name):
    """Return the set of distinct, non-null values in a column.

    A set makes it easy to find the overlap (intersection) between two columns.
    """
    rows = spark.sql(f"SELECT DISTINCT {column_name} FROM {table_name}").collect()
    return {row[0] for row in rows if row[0] is not None}


def overlap_confidence(spark, table1, column1, table2, column2, rows1, rows2):
    """Score how strongly two key columns are related and pick the parent table.

    confidence = shared distinct values / smaller set, as a percentage
    (100% means every value on the smaller side matches -> a clean FK).
    The BIGGER table (fact/hub) becomes the parent, the smaller one the child.
    Returns (confidence_percent, parent_table, child_table, join_key);
    confidence is 0 if either column has no values.
    """
    values1 = get_distinct_values(spark, table1, column1)
    values2 = get_distinct_values(spark, table2, column2)
    if not values1 or not values2:  # nothing to compare -> not a relationship
        return 0.0, table1, table2, column1

    confidence = len(values1 & values2) / min(len(values1), len(values2))

    # bigger table is the parent hub; join key name is the same on both sides
    parent_table, child_table = (table1, table2) if rows1 >= rows2 else (table2, table1)
    join_key = column1
    return round(confidence * 100, 2), parent_table, child_table, join_key


def save_relationship(parent_table, child_table, join_key, confidence):
    """Insert one discovered relationship into MySQL relationship_context."""
    execute(
        "INSERT INTO relationship_context "
        "(parent_table, child_table, join_key, confidence) "
        "VALUES (%s, %s, %s, %s)",
        (parent_table, child_table, join_key, confidence),
    )


def discover():
    """Find all table relationships and store them in MySQL.

    List tables -> collect their key columns + row counts (one scan each) ->
    clear old rows -> compare every table pair on same-name/same-type keys and
    save any real overlaps.
    """
    spark = get_spark()
    spark.sql(f"USE {HIVE_DATABASE}")

    tables = [row.tableName for row in spark.sql("SHOW TABLES").collect()]

    # remember each table's key columns and row count (scan each table once)
    key_columns_by_table = {t: get_key_columns(spark, t) for t in tables}
    row_count_by_table = {t: get_row_count(spark, t) for t in tables}

    execute("DELETE FROM relationship_context")  # start fresh every run
    print("\n=== Discovering relationships ===")

    for table1, table2 in combinations(tables, 2):
        for column1, type1 in key_columns_by_table[table1]:
            for column2, type2 in key_columns_by_table[table2]:
                # only compare columns with the same name AND same type
                if column1.lower() != column2.lower() or type1 != type2:
                    continue

                confidence, parent, child, key = overlap_confidence(
                    spark, table1, column1, table2, column2,
                    row_count_by_table[table1], row_count_by_table[table2],
                )

                if confidence > 0:  # only keep pairs that actually share values
                    save_relationship(parent, child, key, confidence)
                    print(f"  {parent}.{key} <-> {child}.{key}  confidence={confidence}%")

    spark.stop()
    print("=== Relationship discovery complete ===")


if __name__ == "__main__":
    discover()
