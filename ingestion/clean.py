import os
import re
import sys
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, NumericType, IntegralType
from config.config import HDFS_RAW_DIR, HDFS_CLEANED_DIR, HIVE_METASTORE_URI, DATA_DIR


def get_spark():
    return (
        SparkSession.builder
        .appName("TarkAI-Cleaning")
        .config("hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )

def get_string_columns(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]


def get_numeric_columns(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

SQL_RESERVED = {
    "select", "from", "where", "group", "order", "by", "table", "index",
    "date", "time", "timestamp", "user", "case", "when", "then", "end",
    "int", "double", "string", "desc", "asc", "join", "on", "as", "all",
    "and", "or", "not", "null", "like", "in", "is", "count", "sum", "avg",
}

def normalize_column_names(df, report):
    renamed = 0
    seen = {}

    for c in df.columns:
        new_c = c.strip().lower()
        new_c = re.sub(r"[^\w]+", "_", new_c)     
        new_c = re.sub(r"_+", "_", new_c)          
        new_c = new_c.strip("_")

        if new_c == "":                           
            new_c = "col"
        if new_c[0].isdigit():                    
            new_c = "c_" + new_c
        if new_c in SQL_RESERVED:
            new_c = new_c + "_col"

        if new_c in seen:
            seen[new_c] = seen[new_c] + 1
            new_c = f"{new_c}_{seen[new_c]}"
        else:
            seen[new_c] = 0

        if new_c != c:
            df = df.withColumnRenamed(c, new_c)
            renamed += 1

    report["steps"].append({"step": "Normalize Column Names", "columns_renamed": renamed})
    print(f"  Step 1: Renamed {renamed} columns to snake_case")
    return df


DATE_FORMATS = [
    "yyyy-MM-dd",
    "dd-MM-yyyy",
    "dd/MM/yyyy",
    "MM/dd/yyyy",
    "yyyy/MM/dd",
    "dd-MMM-yyyy",
    "yyyy-MM-dd HH:mm:ss",
]

DATE_HINT = re.compile(r"(date|_dt$|_at$|time|dob|birth|created|updated|joined|expiry)")

KEEP_AS_TEXT = re.compile(r"(_id$|^id$|pin|zip|phone|mobile|aadhaar|account|pan|gst)")


def fix_data_types(df, report):
    """Step 2: Turn text columns into real date / number columns."""
    converted = {}

    for c in get_string_columns(df):
        if KEEP_AS_TEXT.search(c):
            continue

        if DATE_HINT.search(c):
            for fmt in DATE_FORMATS:
                parsed = F.to_timestamp(F.col(c), fmt)
                ok = df.select(F.avg(parsed.isNotNull().cast("double"))).first()[0]
                ok = ok if ok is not None else 0
                if ok > 0.90:                      
                    df = df.withColumn(c, parsed)
                    converted[c] = f"timestamp using {fmt}"
                    print(f"  Step 2: '{c}' converted to timestamp ({fmt})")
                    break
            continue

        cleaned = F.regexp_replace(F.col(c), r"[\u20b9$\u20ac\u00a3,%\s]", "")
        as_num = cleaned.cast("double")
        ok = df.select(F.avg(as_num.isNotNull().cast("double"))).first()[0]
        ok = ok if ok is not None else 0
        if ok > 0.95:
            df = df.withColumn(c, as_num)
            converted[c] = "double (removed currency symbols/commas)"
            print(f"  Step 2: '{c}' converted to number")

    report["steps"].append({"step": "Fix Data Types", "converted": converted})
    if not converted:
        print("  Step 2: No type changes needed")
    return df

def remove_duplicates(df, before, report, pk=None, order_col=None):
    if pk:
        if order_col:
            w = Window.partitionBy(*pk).orderBy(F.col(order_col).desc())
            df = df.withColumn("_rn", F.row_number().over(w))
            df = df.filter("_rn = 1").drop("_rn")
        else:
            df = df.dropDuplicates(pk)
    else:
        df = df.dropDuplicates()

    removed = before - df.count()
    report["steps"].append({
        "step": "Remove Duplicates",
        "removed": removed,
        "key_used": pk if pk else "all columns",
    })
    print(f"  Step 3: Removed {removed} duplicate rows")
    return df

def clean_string_columns(df, report):
    string_cols = get_string_columns(df)
    for name in string_cols:
        df = df.withColumn(name, F.trim(F.col(name)))
        df = df.withColumn(name, F.regexp_replace(F.col(name), r"\s+", " "))
        df = df.withColumn(
            name,
            F.when(F.col(name) == "", None).otherwise(F.col(name)),
        )

    report["steps"].append({"step": "Trim Strings", "columns_affected": len(string_cols)})
    print(f"  Step 4: Trimmed {len(string_cols)} string columns")
    return df

CANONICAL_MAPS = {
    "state": {
        "mh": "Maharashtra", "maharastra": "Maharashtra",
        "ka": "Karnataka", "karnatka": "Karnataka",
        "tn": "Tamil Nadu", "dl": "Delhi", "up": "Uttar Pradesh",
        "gj": "Gujarat", "wb": "West Bengal", "ts": "Telangana",
    },
    "gender": {"m": "Male", "f": "Female", "male": "Male", "female": "Female"},
    "status": {"y": "Active", "n": "Inactive", "a": "Active", "i": "Inactive"},
}


def standardize_categories(df, report, max_distinct=100):
    changed = {}

    for c in get_string_columns(df):
        n_distinct = df.select(c).distinct().limit(max_distinct + 1).count()
        if n_distinct > max_distinct:
            continue

        mapping = CANONICAL_MAPS.get(c)
        if mapping:
            expr = F.col(c)
            for wrong, correct in mapping.items():
                expr = F.when(F.lower(F.trim(F.col(c))) == wrong, correct).otherwise(expr)
            df = df.withColumn(c, expr)
            changed[c] = f"{len(mapping)} short forms mapped"
        else:
            original = F.col(c)
            is_lower = original == F.lower(original)
            is_shout = (original == F.upper(original)) & (F.length(original) > 4)
            df = df.withColumn(
                c,
                F.when(is_lower | is_shout, F.initcap(original)).otherwise(original),
            )
            changed[c] = "title case (acronyms preserved)"

    report["steps"].append({"step": "Standardize Categories", "columns": changed})
    print(f"  Step 5: Standardized {len(changed)} category columns")
    return df

#range rules
COLUMN_RULES = {
    "age":              {"min": 0,   "max": 130},
    "dependents":       {"min": 0,   "max": 20},
    "experience_years": {"min": 0,   "max": 70},
    "height_cm":        {"min": 30,  "max": 280},
    "weight_kg":        {"min": 1,   "max": 500},

    # Money 
    "salary":           {"min": 0},
    "bonus":            {"min": 0},
    "unit_price":       {"min": 0},
    "cost":             {"min": 0},
    "sales_amount":     {"min": 0},
    "revenue":          {"min": 0},
    "tax":              {"min": 0},
    "shipping_cost":    {"min": 0},
    "refund_amount":    {"min": 0},

    #Quantities / counts
    "quantity":         {"min": 1},
    "stock":            {"min": 0},
    "reorder_level":    {"min": 0},
    "units_sold":       {"min": 0},
    "clicks":           {"min": 0},
    "impressions":      {"min": 0},
    "page_views":       {"min": 0},
    "num_orders":       {"min": 0},

    #  Rates, ratios, percentages 
    "discount":         {"min": 0,   "max": 100},
    "discount_rate":    {"min": 0,   "max": 1},
    "tax_rate":         {"min": 0,   "max": 1},
    "interest_rate":    {"min": 0,   "max": 100},
    "conversion_rate":  {"min": 0,   "max": 1},
    "churn_rate":       {"min": 0,   "max": 1},
    "margin_pct":       {"min": -100, "max": 100},
    "utilization":      {"min": 0,   "max": 1},
    "completion_pct":   {"min": 0,   "max": 100},
    "probability":      {"min": 0,   "max": 1},

    # Scores / ratings
    "rating":           {"min": 0,   "max": 5},
    "score":            {"min": 0,   "max": 100},
    "csat":             {"min": 1,   "max": 5},
    "credit_score":     {"min": 300, "max": 900},

    # Time / duration 
    "duration_minutes": {"min": 0,   "max": 1440},
    "duration_seconds": {"min": 0},
    "hours_worked":     {"min": 0,   "max": 24},
    "weekly_hours":     {"min": 0,   "max": 168},
    "delivery_days":    {"min": 0,   "max": 365},
    "response_time_ms": {"min": 0},
    "year":             {"min": 1900, "max": 2100},
    "month":            {"min": 1,   "max": 12},
    "day":              {"min": 1,   "max": 31},
    "hour":             {"min": 0,   "max": 23},
    "week":             {"min": 1,   "max": 53},
    "quarter":          {"min": 1,   "max": 4},

    # Geo 
    "latitude":         {"min": -90,  "max": 90},
    "longitude":        {"min": -180, "max": 180},
    "distance_km":      {"min": 0},

    #Sensors / ops 
    "temperature_c":    {"min": -90, "max": 60},
    "humidity_pct":     {"min": 0,   "max": 100},
    "battery_pct":      {"min": 0,   "max": 100},
    "cpu_pct":          {"min": 0,   "max": 100},
}

#junk values
SENTINEL_VALUES = [-1, -999, -9999, 999, 9999, 99999]


def flag_range_violations(df, report):
    summary = {}
    checks = {}

    for column, rule in COLUMN_RULES.items():
        if column not in df.columns:
            continue
        low = rule.get("min")
        high = rule.get("max")

        bad = None
        if low is not None:
            bad = F.col(column) < low
        if high is not None:
            bad = (F.col(column) > high) if bad is None else (bad | (F.col(column) > high))
        if bad is not None:
            bad = bad | F.col(column).isin(SENTINEL_VALUES)
            checks[column] = bad

    if checks:
        counts = df.select(
            [F.sum(p.cast("int")).alias(c) for c, p in checks.items()]
        ).collect()[0].asDict()

        for column, bad in checks.items():
            n = counts[column] if counts[column] else 0
            if n > 0:
                df = df.withColumn(column, F.when(bad, None).otherwise(F.col(column)))
                summary[column] = {
                    "invalid_values": n,
                    "lower_bound": COLUMN_RULES[column].get("min"),
                    "upper_bound": COLUMN_RULES[column].get("max"),
                }
                print(f"  Step 6: Flagged {n} out-of-range values in '{column}'")

    report["steps"].append({
        "step": "Range Validation (rule based)",
        "columns_checked": len(checks),
        "violations": summary,
    })
    if not summary:
        print("  Step 6: No out-of-range values found")
    return df


def audit_nulls(df, report):
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        report["steps"].append({"step": "Null Audit", "null_counts": {}})
        print("  Step 7: No numeric columns to check")
        return df

    row = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols]
    ).collect()[0].asDict()

    nulls = {c: n for c, n in row.items() if n and n > 0}
    for c, n in nulls.items():
        print(f"  Step 7: '{c}' has {n} nulls (will be imputed)")

    report["steps"].append({
        "step": "Null Audit",
        "strategy": "counts recorded before imputation (see Impute Numeric Nulls)",
        "null_counts": nulls,
    })
    if not nulls:
        print("  Step 7: No numeric nulls found")
    return df

def _fill_value_for_column(df, column, strategy):
    """Compute the chosen measure of central tendency for one column."""
    if strategy == "mean":
        return df.select(F.avg(F.col(column))).first()[0]
    if strategy == "mode":
        row = (
            df.filter(F.col(column).isNotNull())
              .groupBy(column)
              .count()
              .orderBy(F.desc("count"), F.asc(column))
              .first()
        )
        return row[column] if row else None
    # default: median via approximate quantile (small relativeError)
    q = df.approxQuantile(column, [0.5], 0.001)
    return q[0] if q else None


def impute_numeric_nulls(df, report, strategy="median"):
    strategy = (strategy or "median").lower()
    if strategy not in ("median", "mean", "mode"):
        print(f"  Step 7b: unknown strategy '{strategy}', falling back to median")
        strategy = "median"

    imputed = {}

    for field in df.schema.fields:
        c = field.name
        if not isinstance(field.dataType, NumericType):
            continue
        if KEEP_AS_TEXT.search(c):
            continue  

        null_count = df.select(F.sum(F.col(c).isNull().cast("int"))).first()[0] or 0
        if null_count == 0:
            continue

        value = _fill_value_for_column(df, c, strategy)
        if value is None:
            imputed[c] = {"nulls_filled": 0, "note": "column fully null, skipped"}
            print(f"  Step 7b: '{c}' is fully null -- nothing to impute from")
            continue

        if isinstance(field.dataType, IntegralType):
            value = int(round(value))
        else:
            value = round(float(value), 4)

        try:
            df = df.na.fill({c: value})
        except Exception as e:
            imputed[c] = {"nulls_filled": 0, "error": str(e)}
            print(f"  (warning) could not impute '{c}': {e}")
            continue

        imputed[c] = {
            "strategy": strategy,
            "fill_value": value,
            "nulls_filled": int(null_count),
        }
        print(f"  Step 7b: '{c}' filled {null_count} nulls with {strategy} = {value}")

    report["steps"].append({
        "step": "Impute Numeric Nulls",
        "strategy": strategy,
        "columns_imputed": imputed,
    })
    if not imputed:
        print("  Step 7b: No numeric nulls to impute")
    return df

NAME_HINT = re.compile(r"(^names?$|^names?_|_names?$|_names?_|first_name|last_name|middle_name|full_name|surname)")


def impute_categorical_nulls(df, report, strategy="all", placeholder="Unknown", max_distinct=100):
    strategy = (strategy or "all").lower()
    if strategy not in ("all", "category", "skip"):
        print(f"  Step 7c: unknown strategy '{strategy}', falling back to all")
        strategy = "all"

    if strategy == "skip":
        report["steps"].append({
            "step": "Impute Text/Categorical Nulls",
            "strategy": "skip",
            "columns_imputed": {},
        })
        print("  Step 7c: Skipped (text nulls kept as NULL)")
        return df

    imputed = {}
    left_untouched = []

    for name in get_string_columns(df):
        if KEEP_AS_TEXT.search(name):
            continue 

        null_count = df.select(F.sum(F.col(name).isNull().cast("int"))).first()[0] or 0
        if null_count == 0:
            continue

        label = "placeholder"
        if strategy == "category":
            n_distinct = df.select(name).distinct().limit(max_distinct + 1).count()
            if n_distinct > max_distinct:
                if NAME_HINT.search(name):
                    label = "placeholder (name column)"
                else:
                    left_untouched.append(name) 
                    print(f"  Step 7c: '{name}' looks like free text -- left untouched")
                    continue

        df = df.na.fill({name: placeholder})
        imputed[name] = {
            "strategy": label,
            "fill_value": placeholder,
            "nulls_filled": int(null_count),
        }
        print(f"  Step 7c: '{name}' filled {null_count} nulls with '{placeholder}'")

    report["steps"].append({
        "step": "Impute Text/Categorical Nulls",
        "strategy": strategy,
        "placeholder": placeholder,
        "columns_imputed": imputed,
        "left_untouched": left_untouched,
    })
    if not imputed:
        print("  Step 7c: No category nulls to impute")
    return df

def build_schema_catalog(df, table_name, max_distinct=50):
    catalog = {
        "table": table_name,
        "row_count": df.count(),
        "columns": [],
    }

    for f in df.schema.fields:
        c = f.name
        entry = {"name": c, "type": f.dataType.simpleString()}

        if isinstance(f.dataType, NumericType):
            stats = df.select(
                F.min(c).alias("min"),
                F.max(c).alias("max"),
                F.round(F.avg(c), 2).alias("mean"),
                F.sum(F.col(c).isNull().cast("int")).alias("nulls"),
            ).collect()[0].asDict()
            entry.update(stats)

        elif isinstance(f.dataType, StringType):
            n = df.select(c).distinct().limit(max_distinct + 1).count()
            entry["distinct_count"] = n
            if n <= max_distinct:
                vals = df.select(c).distinct().collect()
                entry["allowed_values"] = sorted([r[0] for r in vals if r[0] is not None])
            else:
                entry["examples"] = [r[0] for r in df.select(c).limit(3).collect()]

        catalog["columns"].append(entry)

    catalog["sample_rows"] = [r.asDict() for r in df.limit(3).collect()]
    print(f"  Step 8: Catalog built for {len(catalog['columns'])} columns")
    return catalog


#save
def save_cleaned_data(df, table_name, spark=None):
    cleaned_path = f"{HDFS_CLEANED_DIR}/{table_name}"

    df.write.mode("overwrite").parquet(cleaned_path)

    print(f"  Cleaned data saved to: {cleaned_path}")


def save_json(obj, table_name, suffix):
    """Save a dict as JSON inside the local data folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{table_name}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved: {path}")

def clean_dataset(table_name, spark=None, pk=None, order_col=None,
                  impute_strategy="median", categorical_strategy="all",
                  categorical_placeholder="Unknown"):
    own_session = False
    if spark is None:
        spark = get_spark()
        own_session = True

    raw_path = f"{HDFS_RAW_DIR}/{table_name}"
    df = spark.read.option("header", True).option("inferSchema", True).csv(raw_path)
    df = df.cache()

    print(f"\nCleaning: {table_name}")
    before = df.count()
    report = {
        "table": table_name,
        "rows_before": before,
        "columns": len(df.columns),
        "steps": [],
    }
    df = normalize_column_names(df, report)
    df = fix_data_types(df, report)
    df = remove_duplicates(df, before, report, pk=pk, order_col=order_col)
    df = clean_string_columns(df, report)
    df = standardize_categories(df, report)
    df = flag_range_violations(df, report)
    df = audit_nulls(df, report)
    df = impute_numeric_nulls(df, report, strategy=impute_strategy)
    df = impute_categorical_nulls(df, report, strategy=categorical_strategy,
                                  placeholder=categorical_placeholder)

    df = df.cache()
    catalog = build_schema_catalog(df, table_name)

    after = df.count()
    report["rows_after"] = after
    print(f"\n  Rows before: {before} | after: {after} | removed: {before - after}")

    save_cleaned_data(df, table_name, spark)
    save_json(report, table_name, "quality_report")
    save_json(catalog, table_name, "catalog")

    df.unpersist()
    if own_session:
        spark.stop()
    return table_name

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit ingestion/clean.py <table_name> [primary_key]")
        print("       (optional) set IMPUTE_STRATEGY=median|mean|mode to choose the")
        print("       measure of central tendency used to fill numeric nulls")
        print("       (optional) set CATEGORICAL_STRATEGY=all|category|skip to")
        print("       choose how text/categorical nulls are labelled ('Unknown')")
        sys.exit(1)

    table = sys.argv[1]
    key = [sys.argv[2]] if len(sys.argv) > 2 else None
    strategy = os.environ.get("IMPUTE_STRATEGY", "median")
    cat_strategy = os.environ.get("CATEGORICAL_STRATEGY", "all")
    cat_placeholder = os.environ.get("CATEGORICAL_PLACEHOLDER", "Unknown")
    clean_dataset(table, pk=key, impute_strategy=strategy,
                  categorical_strategy=cat_strategy,
                  categorical_placeholder=cat_placeholder)


