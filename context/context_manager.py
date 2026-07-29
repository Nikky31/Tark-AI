# context/context_manager.py
# ------------------------------------------------------------------
# Layer 5: Build and store Schema, Business, and KPI context in MySQL
#
# Business terms are AUTO-DISCOVERED from column names and data types
# instead of being hardcoded - this makes it truly self-service.
# ------------------------------------------------------------------

from pyspark.sql import SparkSession
from config.config import HIVE_DATABASE, HIVE_METASTORE_URI
from config.db import execute, fetch_all


# ---- auto-discovery patterns ----
# maps column name keywords to aggregation functions
# covers multiple domains: sales, HR, healthcare, education, logistics etc
# if a numeric column contains any of these, it becomes a business term
# for any column that doesnt match, we default to SUM anyway
COMMON_METRICS = {
    # sales / finance
    "amount": "SUM",   "price": "SUM",    "cost": "SUM",
    "revenue": "SUM",  "sales": "SUM",    "profit": "SUM",
    "income": "SUM",   "expense": "SUM",  "quantity": "SUM",
    "qty": "SUM",      "total": "SUM",    "discount": "SUM",
    "tax": "SUM",      "fee": "SUM",      "payment": "SUM",
    "balance": "SUM",  "value": "SUM",    "margin": "AVG",
    # HR / people
    "salary": "SUM",   "wage": "SUM",     "bonus": "SUM",
    "experience": "AVG", "tenure": "AVG", "age": "AVG",
    "attendance": "AVG", "hours": "SUM",  "overtime": "SUM",
    # healthcare
    "dosage": "AVG",   "weight": "AVG",   "height": "AVG",
    "bmi": "AVG",      "pressure": "AVG", "pulse": "AVG",
    "temperature": "AVG", "duration": "AVG",
    # education
    "score": "AVG",    "marks": "AVG",    "grade": "AVG",
    "percentage": "AVG", "gpa": "AVG",    "credits": "SUM",
    # ratings / metrics
    "rating": "AVG",   "count": "COUNT",  "frequency": "SUM",
    "distance": "SUM", "speed": "AVG",    "area": "SUM",
    "volume": "SUM",   "units": "SUM",    "inventory": "SUM",
    "stock": "SUM",    "demand": "SUM",   "supply": "SUM",
    # generic
    "number": "SUM",   "size": "AVG",     "length": "AVG",
    "capacity": "SUM", "utilization": "AVG",
}

# numeric types in Hive/Spark
NUMERIC_TYPES = {
    "int", "bigint", "smallint", "tinyint",
    "double", "float", "decimal", "long",
}


def get_spark():
    """Create (or reuse) a Spark session that can talk to Hive."""
    return (
        SparkSession.builder
        .appName("TarkAI-Context")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )


def is_id_or_key_column(col_lower):
    """Return True for ID / key / code columns (these are not metrics)."""
    return any(col_lower.endswith(suffix) for suffix in ("_id", "_key", "_code"))


def is_numeric_type(data_type):
    """Return True if the Hive/Spark data type is numeric (handles decimal(10,2))."""
    base_type = data_type.split("(")[0].lower()
    return base_type in NUMERIC_TYPES


def find_aggregation(col_lower):
    """Pick an aggregation for a column by matching metric keywords (default SUM)."""
    for keyword, agg in COMMON_METRICS.items():
        if keyword in col_lower:
            return agg
    return "SUM"  # default for unknown numeric columns


def build_schema_context(spark):
    """Scan all Hive tables and store their columns + datatypes in MySQL."""
    spark.sql(f"USE {HIVE_DATABASE}")
    tables = [r.tableName for r in spark.sql("SHOW TABLES").collect()]

    execute("DELETE FROM schema_context")
    for t in tables:
        df = spark.table(t)
        for f in df.schema.fields:
            execute(
                "INSERT INTO schema_context (table_name, column_name, data_type) "
                "VALUES (%s, %s, %s)",
                (t, f.name, f.dataType.simpleString()),
            )
    print(f"[Context] Schema built for tables: {tables}")
    return tables


def build_business_context(spark):
    """
    Auto-discover business terms from column names + data types.
    For each numeric column, check if its name matches known metric keywords.
    This way any new dataset automatically gets business term mappings.
    """
    rows = fetch_all("SELECT table_name, column_name, data_type FROM schema_context")
    execute("DELETE FROM business_context")

    mapped = 0
    for table_name, column_name, data_type in rows:
        col_lower = column_name.lower()

        # skip ID and key columns - they are not metrics
        if is_id_or_key_column(col_lower):
            continue

        # only consider numeric columns
        if not is_numeric_type(data_type):
            continue

        # match against known metric patterns (falls back to SUM)
        matched_agg = find_aggregation(col_lower)

        term = col_lower.replace("_", " ")
        execute(
            "INSERT INTO business_context "
            "(business_term, table_name, column_name, aggregation) "
            "VALUES (%s, %s, %s, %s)",
            (term, table_name, column_name, matched_agg),
        )
        mapped += 1

    print(f"[Context] Business terms auto-discovered: {mapped} terms mapped")


def build_kpi_context():
    """Create KPI expressions from the business context."""
    bc = fetch_all("SELECT business_term, table_name, column_name, aggregation FROM business_context")
    execute("DELETE FROM kpi_context")
    for term, table_name, column_name, agg in bc:
        if agg == "COUNT_DISTINCT":
            expr = f"COUNT(DISTINCT {column_name})"
        else:
            expr = f"{agg}({column_name})"
        kpi_name = f"Total {term.title()}"
        execute(
            "INSERT INTO kpi_context (kpi_name, table_name, expression) "
            "VALUES (%s, %s, %s)",
            (kpi_name, table_name, expr),
        )
    print("[Context] KPI context built")


def build_all():
    """Run the full context layer: schema -> business -> KPI."""
    spark = get_spark()
    build_schema_context(spark)
    build_business_context(spark)
    build_kpi_context()
    spark.stop()
    print("\n=== Context layer complete ===")


if __name__ == "__main__":
    build_all()
