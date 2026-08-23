import re
import difflib

from config.db import fetch_all
try:
    from query_engine.rule_engine import ENGLISH_SYNONYMS
    try:
        from query_engine.rule_engine import DIMENSION_HINTS
    except ImportError:
        from query_engine.rule_engine import DIM_HINTS as DIMENSION_HINTS
except Exception as error: 
    print(f"[Validate] Could not import rule engine vocab ({error}); "
          f"falling back to schema/business terms only.")
    ENGLISH_SYNONYMS, DIMENSION_HINTS = {}, {}

STOP_WORDS = {
    "show", "give", "list", "find", "display", "tell", "please", "about",
    "between", "across", "over", "with", "from", "that", "this", "have",
    "want", "need", "would", "could", "there", "their", "where", "which",
    "what", "whats", "were", "does", "much", "many",
}

NUMERIC_TYPES = {
    "int", "bigint", "smallint", "tinyint",
    "double", "float", "decimal", "long",
}

METRIC_KEYWORDS = {
    "revenue", "profit", "loss", "sales", "amount", "cost", "price",
    "income", "earning", "earnings", "margin", "turnover", "spend",
    "expense", "salary", "budget", "payment", "discount", "fee",
    "balance", "wage", "bonus", "tax",

    "quantity", "qty", "count", "total", "number", "value", "volume",
    "units", "inventory", "stock", "demand", "supply", "rating", "score",
    "marks", "gpa", "credits", "attendance", "hours", "overtime",
    "distance", "speed", "duration", "dosage", "weight", "height", "bmi",
    "frequency", "utilization", "capacity",
}

TIME_WORDS = {"time", "date", "day", "week", "month", "quarter", "year",
              "hour", "minute", "period",
              "daily", "weekly", "monthly", "quarterly", "yearly",
              "annual", "annually", "hourly"}

DIMENSION_KEYWORDS = {
    "region", "area", "zone", "territory", "country", "state", "city",
    "town", "district", "branch", "store", "outlet", "warehouse",
    "customer", "client", "buyer", "product", "item", "category", "brand",
    "department", "team", "employee", "staff", "agent", "channel",
    "segment", "supplier", "vendor", "patient", "doctor", "hospital",
    "ward", "clinic", "student", "school", "course", "subject", "grade",
    "route", "vehicle", "driver", "airport", "stadium", "name", "type",
    "group", "gender", "status",
} | TIME_WORDS

WISE_SUFFIXES = ("wise",)
LISTING_KEYWORDS = {"list", "enumerate"}


#from mysql 22/06/26
def load_schema():
    rows = fetch_all(
        "SELECT table_name, column_name, data_type FROM schema_context"
    )
    schema = {}
    all_columns = []
    column_types = {}
    for table_name, column_name, data_type in rows:
        schema.setdefault(table_name, []).append(column_name)
        all_columns.append(column_name)
        column_types[column_name.lower()] = str(data_type).lower()
    return schema, all_columns, column_types


def load_business_terms():
    rows = fetch_all("SELECT DISTINCT business_term FROM business_context")
    return [row[0] for row in rows]

def is_id_or_key(name):
    low = name.lower()
    return any(low.endswith(s) or low == s for s in ("_id", "id", "_key",
                                                     "key", "_code", "code"))


def is_numeric_type(data_type):
    base = data_type.split("(")[0].lower()
    return base in NUMERIC_TYPES


def add_word_and_parts(word, bucket, skip_parts=None):
    low = word.lower()
    bucket.add(low)
    for part in low.replace("_", " ").split():
        if len(part) >= 3 and (skip_parts is None or part not in skip_parts):
            bucket.add(part)


def build_metric_and_dimension_sets(schema, all_columns, column_types,
                                    business_terms):
    metrics = set()
    dimensions = set()
    metric_stems = set(METRIC_KEYWORDS)

    for w in list(METRIC_KEYWORDS):
        if w.endswith("s"):
            metric_stems.add(w[:-1])  
        else:
            metric_stems.add(w + "s")  

    for column_name in all_columns:
        if is_id_or_key(column_name):
            continue  
        data_type = column_types.get(column_name.lower(), "string")
        if is_numeric_type(data_type):
            add_word_and_parts(column_name, metrics)
        else:
            add_word_and_parts(column_name, dimensions, skip_parts=metric_stems)

    for term in business_terms:
        add_word_and_parts(term, metrics)

    for key, synonyms in ENGLISH_SYNONYMS.items():
        words = [key] + list(synonyms)
        if key.lower() in METRIC_KEYWORDS:
            for w in words:
                metrics.add(w.lower())
        elif key.lower() in DIMENSION_KEYWORDS:
            for w in words:
                dimensions.add(w.lower())

    for key, hints in DIMENSION_HINTS.items():
        dimensions.add(key.lower())
        for hint in hints:
            dimensions.add(hint.lower())

    metrics |= METRIC_KEYWORDS
    dimensions |= DIMENSION_KEYWORDS

    for w in list(metrics & dimensions):
        if w in TIME_WORDS:
            metrics.discard(w)
        else:
            dimensions.discard(w)

    def clean(s):
        return {w for w in s if w and not w.isdigit() and w not in {"id", "key"}}

    return clean(metrics), clean(dimensions)


def build_vocabulary(metrics, dimensions):
    return metrics | dimensions

def strip_wise(token):
    for suffix in WISE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return None


def get_query_word_forms(query):
    tokens = set(re.findall(r"[a-z0-9_]+", query))
    forms = set(tokens)
    for token in tokens:
        # singular/plural
        if token.endswith("s") and len(token) > 3:
            forms.add(token[:-1])  
        elif len(token) >= 3:
            forms.add(token + "s")  
        base = strip_wise(token)
        if base:
            forms.add(base)
            if base.endswith("s") and len(base) > 3:
                forms.add(base[:-1])
    return tokens, forms


def find_matches(word_forms, vocabulary):
    found = set()
    for term in vocabulary:
        if " " in term:            
            continue
        if term in word_forms:
            found.add(term)
    return found


def referenced_metrics_and_dimensions(query, metrics, dimensions):
    _, word_forms = get_query_word_forms(query)

    found_metrics = find_matches(word_forms, metrics)
    found_dimensions = find_matches(word_forms, dimensions)

    for term in metrics:
        if " " in term and term in query:
            found_metrics.add(term)
    for term in dimensions:
        if " " in term and term in query:
            found_dimensions.add(term)

    return sorted(found_metrics), sorted(found_dimensions)


def suggest_correction(word, known_words):
    matches = difflib.get_close_matches(word, known_words, n=1, cutoff=0.6)
    return matches[0] if matches else None


def is_listing_query(query):
    words = set(re.findall(r"[a-z]+", query.lower()))
    return bool(words & LISTING_KEYWORDS)

def validate_query(query):
    schema, all_columns, column_types = load_schema()
    business_terms = load_business_terms()

    if not schema:
        return False, "No tables found. Please upload a dataset first."

    lower_query = query.lower()
    metrics, dimensions = build_metric_and_dimension_sets(
        schema, all_columns, column_types, business_terms
    )

    found_metrics, found_dimensions = referenced_metrics_and_dimensions(
        lower_query, metrics, dimensions
    )

    has_metric = bool(found_metrics)
    has_dimension = bool(found_dimensions)

    if has_metric and has_dimension:
        return True, (f"Valid. Metric: {found_metrics} | "
                      f"Dimension: {found_dimensions}")
    
    if has_dimension and is_listing_query(lower_query):
        return True, (f"Valid (listing query). "
                      f"Dimension: {found_dimensions}")

    if has_metric and not has_dimension:
        #"highest earning" - WHAT should we measure?
        return False, (
            f"Incomplete query: found metric {found_metrics} but no group to "
            f"break it down by. Add a dimension, e.g. \"{found_metrics[0]} by "
            f"region / product / customer / month\"."
        )

    if has_dimension and not has_metric:
        #"by region" / "list customers" - measure WHAT?
        return False, (
            f"Incomplete query: found dimension {found_dimensions} but no "
            f"metric to measure. Add a metric, e.g. \"revenue / profit / count "
            f"by {found_dimensions[0]}\"."
        )
    
    vocabulary = build_vocabulary(metrics, dimensions)
    candidate_words = [
        word for word in re.findall(r"[a-z]+", lower_query)
        if len(word) > 3 and word not in STOP_WORDS
    ]
    candidate_words.sort(key=len, reverse=True)

    for word in candidate_words:
        suggestion = suggest_correction(word, list(vocabulary))
        if suggestion:
            return False, f"Could not match '{word}'. Did you mean '{suggestion}'?"

    return False, ("Query does not reference any known metric or dimension. "
                   "Try naming a value and a group, e.g. \"revenue by region\".")


if __name__ == "__main__":
    #test
    test_queries = [
        
        "total revenue by region",
        "highest earning customer",
        "top customers by sales",
        "revenue yearwise",
        "profit regionwise",
        "sales monthwise",
        "earnings by city",
        "average profit per product",

        
        "highest earning",
        "total revenue",
        "show profit",
        "revanue",                
       

        "list all region",
        "list all regions",


        "by customer",
 
        "xyz abc",
        "hello there",
    ]
    for test_query in test_queries:
        ok, msg = validate_query(test_query)
        mark = "VALID  " if ok else "INVALID"
        print(f"{mark} | {test_query:32} -> {msg}")
