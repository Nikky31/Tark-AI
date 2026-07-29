# query_engine/rule_engine.py
# ---------------------------------------------------------------------
# LAYER 8 : SQL GENERATION (rule-based)
#
# The user asks something like "Top 5 customers with sales more than 5000"
# and Layer 6 already gave us the INTENT (e.g. "Ranking"). Our job is to
# turn that question into real Hive SQL using simple, explainable RULES
# (no AI). If the rules cannot build a query we return None and let the
# Ollama LLM take over.
#
# HOW ONE QUERY BECOMES SQL:
#   1. metric     -> the number to measure   (e.g. SUM(sales_amount))
#   2. dimension  -> the column to GROUP BY  (e.g. customer_name)
#   3. WHERE      -> filters like "region = West"
#   4. HAVING     -> filters on the metric like "more than 5000"
#   5. JOIN       -> if metric and dimension live in different tables
#   6. assemble   -> final shape depends on the intent
#
# FINDING THE METRIC uses a 5-layer strategy (most exact first):
#   1 direct term  2 real column  3 english synonym  4 fuzzy/typo
#   5 default (first SUM metric when the query clearly wants data)
#
# Everything is DYNAMIC: table/column names are never hardcoded, they come
# from the MySQL metadata (business_context, schema_context,
# relationship_context), so this works for ANY loaded dataset.
# ---------------------------------------------------------------------

import sys
import re
import difflib

from config.db import fetch_all
from config.config import HIVE_DATABASE
from nlp.intent import detect_intent

# BUILD MARKER: if evaluate.py does not print this, an old rule_engine.py
# is still on the import path and these fixes are not active.
print("[RuleEngine] BUILD 2026-07-24-FIXES: KPI->Aggregation promote | plural dims | age-guard | default-metric", file=sys.stderr)


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

# rows to return when the user does not say "top 5" / "bottom 3"
DEFAULT_LIMIT = 10

# Hive data types we treat as "numbers" (only these can be a SUM/AVG metric)
NUMERIC_TYPES = {
    "int", "integer", "bigint", "smallint", "tinyint",
    "double", "float", "decimal", "long", "number", "numeric",
}

# UNIVERSAL english synonyms (language facts, not data facts): jump from an
# everyday word to the kind of metric column it probably means.
ENGLISH_SYNONYMS = {
    "sell": ["sale", "sales", "revenue", "amount", "income"],
    "sold": ["sale", "sales", "revenue", "amount"],
    "selling": ["sale", "sales", "revenue"],
    "buy": ["purchase", "quantity", "amount", "order"],
    "bought": ["purchase", "quantity", "amount"],
    "earn": ["revenue", "profit", "income", "salary", "earning"],
    "earned": ["revenue", "profit", "income", "salary"],
    "earnings": ["revenue", "profit", "income", "salary"],
    "spend": ["cost", "expense", "amount", "spending"],
    "spent": ["cost", "expense", "amount"],
    "money": ["revenue", "profit", "income", "salary", "amount", "cost"],
    "income": ["revenue", "profit", "salary", "earning"],
    "cost": ["cost", "price", "expense", "fee"],
    "pay": ["salary", "payment", "amount", "cost"],
    "paid": ["salary", "payment", "amount"],
    "charge": ["fee", "cost", "price", "amount"],
    "perform": ["revenue", "profit", "score", "rating", "amount", "salary"],
    "performer": ["revenue", "profit", "score", "rating", "amount"],
    "performers": ["revenue", "profit", "score", "rating", "amount"],
    "performing": ["revenue", "profit", "score", "rating", "amount"],
    "figure": ["amount", "revenue", "total", "value", "count"],
    "value": ["amount", "value", "price", "score", "revenue"],
    "numbers": ["quantity", "count", "amount", "total", "revenue"],
    "everything": ["amount", "total", "revenue", "count", "quantity"],
    "breakdown": ["amount", "revenue", "profit", "count", "quantity"],
    "analysis": ["amount", "revenue", "profit", "count", "score"],
    "track": ["quantity", "amount", "count", "score"],
    "growth": ["revenue", "profit", "amount", "salary", "score"],
    "pattern": ["revenue", "profit", "amount", "count", "score"],
    "split": ["amount", "revenue", "profit", "count"],
    "visual": ["amount", "revenue", "profit", "count", "quantity"],
    "leading": ["revenue", "profit", "amount", "score", "rating"],
    "seller": ["sale", "sales", "revenue", "quantity"],
    "sellers": ["sale", "sales", "revenue", "quantity"],
    "order": ["order", "quantity", "amount", "count"],
    "orders": ["order", "quantity", "amount"],
    "demand": ["quantity", "order", "amount", "count"],
    # money words -> amount/revenue metric (never a plain count). "amount"
    # still matches a real column like total_amount / sales_amount.
    "profit": ["profit", "revenue", "income", "amount", "margin", "earning"],
    "profits": ["profit", "revenue", "income", "amount", "margin"],
    "margin": ["margin", "profit", "revenue", "amount"],
    "revenue": ["revenue", "sales", "amount", "income", "turnover"],
    "revenues": ["revenue", "sales", "amount", "income"],
    "sales": ["sales", "sale", "revenue", "amount", "turnover"],
    "turnover": ["revenue", "sales", "amount"],
    "gross": ["revenue", "sales", "amount", "profit"],
}

# UNIVERSAL dimension hints: everyday grouping word -> column-name it points at
DIMENSION_HINTS = {
    "city": ["city", "town", "location"],
    "region": ["region", "area", "zone", "state", "territory"],
    "category": ["category", "type", "class", "group", "segment"],
    "customer": ["customer", "client", "buyer", "user", "member"],
    "product": ["product", "item", "good", "sku"],
    "employee": ["employee", "emp", "staff", "worker"],
    "department": ["department", "dept", "division", "unit", "branch"],
    "branch": ["branch", "office", "store", "outlet"],
    "month": ["date", "month", "period", "time"],
    "year": ["date", "year", "period", "time"],
    "quarter": ["date", "quarter", "period"],
    "type": ["type", "category", "kind", "class"],
    "name": ["name"],
    "status": ["status", "state", "condition"],
    "country": ["country", "nation"],
    "gender": ["gender", "sex"],
    "categories": ["category", "type", "class"],
}

# words that show the user really wants data (gate for the default metric)
INTENT_WORDS = {
    "total", "average", "sum", "count", "show", "get", "find",
    "what", "how", "which", "top", "bottom", "best", "worst",
    "compare", "plot", "chart", "graph", "trend", "rank",
    "forecast", "predict", "by", "per", "each", "all",
    "give", "display", "list", "highest", "lowest", "maximum",
    "minimum", "biggest", "smallest", "most", "least",
}

# words that filter the metric or ask for a per-group breakdown
THRESHOLD_WORDS = {
    "more", "than", "exceed", "exceeds", "above", "below", "under",
    "over", "each", "with",
}


# ---------------------------------------------------------------------
# PART 1 : LOAD METADATA FROM MYSQL (stored by earlier layers)
# ---------------------------------------------------------------------

def load_business_terms():
    """business_term -> (table, column, aggregation). Most trusted source."""
    rows = fetch_all(
        "SELECT business_term, table_name, column_name, aggregation "
        "FROM business_context"
    )
    business_terms = {}
    for term, table, column, aggregation in rows:
        business_terms[term.lower()] = (table, column, aggregation)
    return business_terms


def load_schema():
    """table_name -> [column names]."""
    rows = fetch_all("SELECT table_name, column_name FROM schema_context")
    schema = {}
    for table, column in rows:
        schema.setdefault(table, []).append(column)
    return schema


def load_column_types():
    """table_name -> {column_name: data_type}. Lets us tell numbers from text."""
    rows = fetch_all(
        "SELECT table_name, column_name, data_type FROM schema_context"
    )
    column_types = {}
    for table, column, data_type in rows:
        column_types.setdefault(table, {})[column] = data_type
    return column_types


# ---------------------------------------------------------------------
# PART 2 : SMALL REUSABLE HELPERS
# ---------------------------------------------------------------------

def is_id_column(column_name):
    """True if the column looks like an id/key/code (do not show/group/SUM)."""
    name = column_name.lower()
    return (
        name == "id"
        or name.endswith("_id")
        or name.endswith("_key")
        or name.endswith("code")
    )


def is_numeric_column(table, column, column_types):
    """True if the column's Hive type is numeric (handles 'decimal(10,2)')."""
    data_type = column_types.get(table, {}).get(column, "")
    base_type = data_type.split("(")[0].strip().lower()
    return base_type in NUMERIC_TYPES


def guess_aggregation(column_name):
    """Default agg for a numeric column: AVG for rate/rating/score/age, else SUM."""
    name = column_name.lower()
    average_hints = ("rate", "rating", "score", "percent",
                     "ratio", "age", "avg", "average")
    if any(hint in name for hint in average_hints):
        return "AVG"
    return "SUM"


def make_aggregation(aggregation, column):
    """Build aggregation SQL text, e.g. SUM(sales_amount)."""
    if aggregation == "COUNT_DISTINCT":
        return f"COUNT(DISTINCT {column})"
    return f"{aggregation}({column})"


def aggregation_from_verb(query, intent):
    """Override the stored agg when the user's words ask for avg/count/max/min.

    Word boundaries stop "count" firing on "country". highest/lowest mean
    MAX/MIN only for a KPI (in a Ranking they just set the sort order).
    """
    text = query.lower()
    if re.search(r"\b(average|avg|mean)\b", text):
        return "AVG"
    if re.search(r"\b(how many|number of|count)\b", text):
        return "COUNT"
    if re.search(r"\b(maximum|max)\b", text):
        return "MAX"
    if re.search(r"\b(minimum|min)\b", text):
        return "MIN"
    if intent == "KPI":
        if re.search(r"\b(highest|largest|peak|most)\b", text):
            return "MAX"
        if re.search(r"\b(lowest|smallest|least)\b", text):
            return "MIN"
    return None


def is_time_dimension(table, column, column_types):
    """True if a dimension is a date/time (by data type, else by name hints)."""
    data_type = column_types.get(table, {}).get(column, "").lower()
    if "date" in data_type or "time" in data_type or "timestamp" in data_type:
        return True
    name = column.lower()
    time_hints = ("date", "month", "year", "time", "day", "period",
                  "quarter", "week")
    return any(hint in name for hint in time_hints)


def make_time_bucket(dimension_sql, query):
    """Bucket a raw date column into a period (month is the trend default)."""
    text = query.lower()
    if re.search(r"\bquarter", text):
        return (f"concat(cast(year({dimension_sql}) as string), '-Q', "
                f"cast(quarter({dimension_sql}) as string))")
    if re.search(r"\b(year|yearly|annual|annually)\b", text):
        return f"year({dimension_sql})"
    if re.search(r"\b(week|weekly)\b", text):
        return f"date_format({dimension_sql}, 'yyyy-ww')"
    if re.search(r"\b(daily|day)\b", text):
        return f"date_format({dimension_sql}, 'yyyy-MM-dd')"
    return f"date_format({dimension_sql}, 'yyyy-MM')"


def make_alias(term):
    """Business term -> safe SQL alias (drop spaces/symbols)."""
    alias = re.sub(r"\W+", "", term.strip().lower()).strip("_")
    return alias or "value"


def pick_label_column(table, schema):
    """Best display column for an entity table: name col, then id/key, then first."""
    columns = schema.get(table, [])
    for column in columns:
        if column.lower() == "name" or column.lower().endswith("name"):
            return column
    for column in columns:
        if column.lower().endswith("_id") or column.lower().endswith("key"):
            return column
    return columns[0] if columns else None


def word_in_text(word, text):
    """True if `word` appears in `text` as a WHOLE word (avoids age in average)."""
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def singularize(word):
    """Naive plural -> singular so hint tables match ("regions" -> "region")."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"      # categories -> category
    if word.endswith("s") and len(word) > 3:
        return word[:-1]            # regions -> region
    return word


# ---------------------------------------------------------------------
# PART 3 : FIND THE METRIC  (the 5-layer strategy, most exact first)
# ---------------------------------------------------------------------

def find_metric(query, business_terms, schema, column_types):
    """Work out WHICH number the user wants. Stops at the first confident hit.

    Returns (term, (table, column, aggregation)) or (None, None).
    """
    words = query.lower().split()

    # lookup of every column: name -> (table, column)
    all_columns = {}
    for table, columns in schema.items():
        for column in columns:
            all_columns[column.lower()] = (table, column)

    # LAYER 1: business term appears in the query. Keep the LONGEST match
    # ("order value" beats "value").
    direct_matches = []
    for term, info in business_terms.items():
        if word_in_text(term, query):
            direct_matches.append((term, info))
    if direct_matches:
        direct_matches.sort(key=lambda item: len(item[0]), reverse=True)
        best_term, best_info = direct_matches[0]
        print(f"[RuleEngine] Layer 1 direct match: '{best_term}'")
        return best_term, best_info

    # LAYER 2a: a query word matches a column that has a business term.
    for word in words:
        if len(word) < 3:
            continue
        for term, info in business_terms.items():
            column_of_term = info[1].lower()
            if word in column_of_term and not is_id_column(column_of_term):
                print(f"[RuleEngine] Layer 2 column->term: '{word}' -> '{term}'")
                return term, info
    # LAYER 2b: else use the numeric column directly and guess the agg.
    for word in words:
        if len(word) < 3:
            continue
        for column_lower, (table, column) in all_columns.items():
            if is_id_column(column_lower):
                continue
            if word in column_lower and is_numeric_column(table, column, column_types):
                aggregation = guess_aggregation(column)
                print(f"[RuleEngine] Layer 2 dynamic column: '{word}' -> {column}")
                return column, (table, column, aggregation)

    # LAYER 3: expand an everyday word to synonyms, match term or column.
    for word in words:
        for synonym in ENGLISH_SYNONYMS.get(word, []):
            for term, info in business_terms.items():
                if synonym in term or term in synonym:
                    print(f"[RuleEngine] Layer 3 synonym: '{word}' -> '{term}'")
                    return term, info
            for column_lower, (table, column) in all_columns.items():
                if is_id_column(column_lower):
                    continue
                if synonym in column_lower and is_numeric_column(table, column, column_types):
                    # prefer a business term for this column (correct agg)...
                    for term, info in business_terms.items():
                        if info[1].lower() == column_lower:
                            print(f"[RuleEngine] Layer 3 synonym->term: '{word}' -> '{term}'")
                            return term, info
                    # ...else use the numeric column straight from schema.
                    aggregation = guess_aggregation(column)
                    print(f"[RuleEngine] Layer 3 synonym->column: '{word}' -> {column}")
                    return column, (table, column, aggregation)

    # LAYER 4a: partial match ("prof" in "profit"). Reverse match only for
    # terms >= 4 chars so "age" cannot hide inside "aver-age".
    for word in words:
        if len(word) < 4:
            continue
        for term, info in business_terms.items():
            term_nospace = term.replace(" ", "")
            if word in term or (len(term_nospace) >= 4 and term_nospace in word):
                print(f"[RuleEngine] Layer 4 fuzzy match: '{word}' -> '{term}'")
                return term, info
    # LAYER 4b: typo match via edit-distance ("revanue" -> "revenue").
    term_names = list(business_terms.keys())
    for word in words:
        if len(word) < 4:
            continue
        close = difflib.get_close_matches(word, term_names, n=1, cutoff=0.8)
        if close:
            term = close[0]
            print(f"[RuleEngine] Layer 4 typo match: '{word}' -> '{term}'")
            return term, business_terms[term]

    # LAYER 5: default. Only if an intent word is present but no metric named.
    # Prefer a money metric over a plain count, then any SUM, then first term.
    has_intent_word = any(
        word in INTENT_WORDS
        or word in DIMENSION_HINTS
        or singularize(word) in DIMENSION_HINTS
        or word in THRESHOLD_WORDS
        for word in words
    )
    if has_intent_word and business_terms:
        money_hints = ("amount", "revenue", "sales", "total", "price",
                       "profit", "income", "cost", "margin", "turnover", "value")
        sum_terms = [(term, info) for term, info in business_terms.items()
                     if info[2] == "SUM"]
        for term, info in sum_terms:
            column_name = info[1].lower()
            if any(h in column_name or h in term for h in money_hints):
                print(f"[RuleEngine] Layer 5 money default -> '{term}'")
                return term, info
        if sum_terms:
            term, info = sum_terms[0]
            print(f"[RuleEngine] Layer 5 default -> '{term}'")
            return term, info
        first_term = next(iter(business_terms))
        return first_term, business_terms[first_term]

    return None, None


# ---------------------------------------------------------------------
# PART 4 : FIND THE DIMENSION  (the GROUP BY column)
# ---------------------------------------------------------------------

def find_dimension(query, schema, column_types, metric_table, metric_column):
    """Which column to GROUP BY / show with the metric. Most exact rule first.

    Returns (column, table) or (None, metric_table). Never the metric itself.
    """
    text = query.lower()
    words = text.split()

    def usable(column):
        """A dimension must not be an id column or the metric itself."""
        return (
            not is_id_column(column)
            and column.lower() != metric_column.lower()
        )

    def column_mentioned(column):
        """Column named in the query, singular OR plural (optional trailing s)."""
        return re.search(r"\b" + re.escape(column) + r"s?\b", text) is not None

    # STEP 1: the user typed a real column name (metric table first).
    for column in schema.get(metric_table, []):
        if usable(column) and column_mentioned(column.lower()):
            return column, metric_table
    for table, columns in schema.items():
        for column in columns:
            if usable(column) and column_mentioned(column.lower()):
                return column, table

    # STEP 2: dynamic dimension (customer_name -> customer, name).
    dynamic_dimensions = build_dynamic_dimension_map(schema)
    for word in words:
        if len(word) < 3:
            continue
        candidates = (dynamic_dimensions.get(word, [])
                      or dynamic_dimensions.get(singularize(word), []))
        for column, table in candidates:
            if usable(column):
                print(f"[RuleEngine] Dynamic dim: '{word}' -> {column} in {table}")
                return column, table

    # STEP 3: english hint ("city" -> city/town/location column).
    for word in words:
        for key in (word, singularize(word)):
            for hint in DIMENSION_HINTS.get(key, []):
                for table, columns in schema.items():
                    for column in columns:
                        if usable(column) and hint in column.lower():
                            print(f"[RuleEngine] Dim hint: '{word}' -> {column} in {table}")
                            return column, table

    # STEP 3b: the word names a whole table -> use its best label column.
    for word in words:
        singular = word[:-1] if word.endswith("s") and len(word) > 3 else word
        for table in schema:
            if table.lower() in (word, singular):
                label = pick_label_column(table, schema)
                if label:
                    print(f"[RuleEngine] Entity dim: '{word}' -> {label} in {table}")
                    return label, table

    # STEP 4: default -> first text column in the metric table.
    for column in schema.get(metric_table, []):
        if not usable(column):
            continue
        if not is_numeric_column(metric_table, column, column_types):
            return column, metric_table

    return None, metric_table


def build_dynamic_dimension_map(schema):
    """Build word -> [(column, table)] by splitting text column names.

    Lets "customer" find "customer_name" for ANY dataset, zero hardcoding.
    """
    dynamic_dimensions = {}
    for table, columns in schema.items():
        for column in columns:
            if is_id_column(column):
                continue
            parts = column.lower().replace("_", " ").split()
            for part in parts:
                if len(part) < 3:
                    continue
                dynamic_dimensions.setdefault(part, []).append((column, table))
    return dynamic_dimensions


# ---------------------------------------------------------------------
# PART 5 : EXTRA CLAUSES (LIMIT / WHERE / HAVING / COMPARE / JOIN)
# ---------------------------------------------------------------------

def extract_limit(query):
    """How many rows to return.

    Handles the usual "top 5" / "5 best", plus explicit counts like
    "show 8 customers" or "give me 3 cities", and "which ... best" -> 1.
    Numbers that are really thresholds ("more than 5000") or years like
    "2024" are ignored so they never leak into the LIMIT. Falls back to
    DEFAULT_LIMIT when no count is mentioned.
    """
    text = query.lower()

    # "top 5" / "bottom 3"
    match = re.search(r"(?:top|bottom)\s+(\d+)", text)
    if match:
        return int(match.group(1))

    # "5 best" / "3 highest" / "10 largest" ...
    match = re.search(
        r"(\d+)\s+(?:top|highest|lowest|best|worst|largest|smallest)", text
    )
    if match:
        return int(match.group(1))

    # explicit count right after an action/limit word:
    # "show 8 customers", "list 5", "give me 3 cities", "first 20", "only 15"
    match = re.search(
        r"\b(?:show|list|give|display|get|find|fetch|return|first|last|only|just|me)"
        r"\s+(\d+)\b",
        text,
    )
    if match and is_row_count(int(match.group(1))):
        return int(match.group(1))

    # a bare "<number> <plural-noun>" count, e.g. "8 customers by profit".
    # Skip threshold numbers ("more than 5000") and year-like numbers.
    for found in re.finditer(r"(\d[\d,]*)\s+([a-z]+)", text):
        number = int(found.group(1).replace(",", ""))
        following = found.group(2)
        preceding = text[:found.start()]
        is_threshold = re.search(
            r"\b(?:than|over|above|below|under|least|most|exceed|exceeds|at)\s+$",
            preceding,
        )
        if is_threshold or not is_row_count(number):
            continue
        if (following.endswith("s")
                or following in DIMENSION_HINTS
                or singularize(following) in DIMENSION_HINTS):
            return number

    # "which/what/who ... best" -> a single winner
    asks_for_single = re.search(r"\b(which|what|who)\b", text)
    superlative = re.search(
        r"\b(best|worst|highest|lowest|largest|smallest|most|least|top)\b", text
    )
    if asks_for_single and superlative:
        return 1

    return DEFAULT_LIMIT


def is_row_count(number):
    """A sensible row-count limit: a small positive number.

    Rejects thresholds like 5000 and years like 2024 so they are never
    mistaken for "how many rows to show".
    """
    return 0 < number < 1000


# ordinal words -> rank number, so "second highest" becomes rank 2
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

# superlative words that turn an ordinal into a rank ("second HIGHEST")
_RANK_SUPERLATIVE = (
    r"(?:highest|largest|biggest|greatest|top|most|best|"
    r"lowest|smallest|least|worst|maximum|minimum)"
)


def extract_nth_rank(query):
    """Detect "second highest" / "3rd largest" / "second lowest" -> rank N.

    Returns the 1-based rank (2 for "second highest") or None when there is no
    ordinal. The caller fetches exactly that row with ORDER BY ... LIMIT 1
    OFFSET N-1, which needs no sub-query and no window function. Rank 1
    ("first"/"1st") returns None so the ordinary top-1 LIMIT path handles it.
    """
    text = query.lower()
    for word, number in ORDINAL_WORDS.items():
        if re.search(r"\b" + word + r"\b\s+" + _RANK_SUPERLATIVE, text):
            return number if number > 1 else None
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b\s+" + _RANK_SUPERLATIVE, text)
    if match:
        number = int(match.group(1))
        return number if number > 1 else None
    return None


def parse_number(raw_number):
    """Clean a number string: "5,000"->"5000", "5k"->"5000", "2m"->"2000000"."""
    text = raw_number.lower().replace(",", "").strip()
    multiplier = 1
    if text.endswith("k"):
        multiplier, text = 1_000, text[:-1]
    elif text.endswith("m"):
        multiplier, text = 1_000_000, text[:-1]
    value = float(text) * multiplier
    return str(int(value)) if value.is_integer() else str(value)


def extract_where(query, schema):
    """Find a simple equality filter, e.g. "where region = West".

    Returns (column, value, table) or (None, None, None). Uses the ORIGINAL
    query so the value keeps its case ('West' stays 'West').
    """
    patterns = [
        # where region = 'West' / for city is Pune / of type equals A
        r"(?:where|for|in|of)\s+(\w+)\s*(?:=|is|equals?)\s*['\"]?([^'\"]+?)['\"]?(?:\s|$)",
        # region = 'West'
        r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]",
        # where region West   (two bare words at the end)
        r"(?:where|for|in)\s+(\w+)\s+(\w+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        column_hint = match.group(1).lower().strip()
        value = match.group(2).strip()
        for table, columns in schema.items():
            for column in columns:
                if column_hint == column.lower() or column_hint in column.lower():
                    return column, value, table
    return None, None, None


def resolve_where_value(table, column, value):
    """Return the value's REAL stored casing so the filter actually matches.

    Blindly Title-Casing breaks acronyms ("upi" -> "Upi" but the data stores
    "UPI"). Instead we look the value up in the actual table, comparing
    case-insensitively, and use whatever spelling the data really stores. If
    the lookup finds nothing or the executor is unavailable, we fall back to
    Title Case, which is correct for most plain dimension values.
    """
    try:
        from query_engine.executor import lookup_matching_value
        actual_value = lookup_matching_value(table, column, value)
        if actual_value:
            return actual_value
    except Exception as error:
        print(f"[RuleEngine] value lookup unavailable ({error}); using Title Case")
    return value.title()


def extract_having(query, metric_column, aggregation):
    """Filter on the METRIC, e.g. "more than 5000" -> " HAVING SUM(x) > 5000"."""
    patterns = [
        (r"(?:more than|greater than|above|exceed|exceeds|over)\s+([\d,.]+[km]?)", ">"),
        (r"(?:less than|below|under|fewer than)\s+([\d,.]+[km]?)", "<"),
        (r"(?:at least|minimum|min)\s+([\d,.]+[km]?)", ">="),
        (r"(?:at most|maximum|max)\s+([\d,.]+[km]?)", "<="),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            number = parse_number(match.group(1))
            metric_expression = make_aggregation(aggregation, metric_column)
            print(f"[RuleEngine] HAVING: {metric_expression} {operator} {number}")
            return f" HAVING {metric_expression} {operator} {number}"
    return ""


def extract_comparison_values(query):
    """Read the two things in "compare X and Y" / "between X and Y".

    Returns e.g. ["Pune", "Mumbai"] (original case) or [] if not found.
    """
    match = re.search(
        r"(?:between|compare)\s+([\w ]+?)\s+(?:and|vs|versus|,)\s+([\w ]+?)(?:\s|$)",
        query,
        re.IGNORECASE,
    )
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    return []


def find_join(table_one, table_two):
    """Look up how two tables relate via relationship_context.

    Returns (join_sql, parent_table, child_table) or (None, None, None).
    """
    relationships = fetch_all(
        "SELECT parent_table, child_table, join_key FROM relationship_context "
        "WHERE (parent_table=%s AND child_table=%s) "
        "   OR (parent_table=%s AND child_table=%s)",
        (table_one, table_two, table_two, table_one),
    )
    if not relationships:
        return None, None, None

    parent, child, join_key = relationships[0]
    database = HIVE_DATABASE
    join_sql = (
        f"{database}.{parent} JOIN {database}.{child} "
        f"ON {parent}.{join_key} = {child}.{join_key}"
    )
    return join_sql, parent, child


def find_join_key(table_one, table_two):
    """Return the column two tables join on (from relationship_context).

    Returns the join_key string (e.g. "store_id") or None if the two tables
    have no known relationship.
    """
    relationships = fetch_all(
        "SELECT join_key FROM relationship_context "
        "WHERE (parent_table=%s AND child_table=%s) "
        "   OR (parent_table=%s AND child_table=%s)",
        (table_one, table_two, table_two, table_one),
    )
    if relationships:
        return relationships[0][0]
    return None


def find_profit_parts(schema, column_types):
    """Locate the columns needed for real profit, without hardcoding names.

    profit = revenue - quantity * unit_cost, so we look for three numeric
    columns anywhere in the schema:
      - revenue  : a sales / amount column (e.g. sales_amount)
      - quantity : units sold
      - cost     : per-unit cost (e.g. cost_price)
    Returns {"revenue": (table, col), "quantity": (t, c), "cost": (t, c)} or
    None when any piece is missing (caller then keeps the old behaviour).
    """
    revenue = quantity = cost = None
    for table, columns in schema.items():
        for column in columns:
            low = column.lower()
            if is_id_column(low) or not is_numeric_column(table, column, column_types):
                continue
            if revenue is None and (
                "sales_amount" in low or "revenue" in low
                or "turnover" in low or low == "amount" or low.endswith("_amount")
            ):
                revenue = (table, column)
            if quantity is None and (low == "qty" or "quantity" in low):
                quantity = (table, column)
            if cost is None and (
                "cost_price" in low or "unit_cost" in low
                or low == "cost" or low.endswith("_cost") or "cost" in low
            ):
                cost = (table, column)
    if revenue and quantity and cost:
        return {"revenue": revenue, "quantity": quantity, "cost": cost}
    return None


def build_profit_sql(query, schema, column_types, intent, database):
    """Build SQL for a real-profit question (no sub-query, no window function).

    Handles "profit by <dim>", "second highest profit by region", "total
    profit", plus an optional WHERE filter. profit is computed inline as
    SUM(revenue - quantity * cost); the Nth-highest case just skips N-1 rows
    with LIMIT 1 OFFSET N-1. Returns the SQL string, or None to fall back to
    the ordinary metric path when the dataset lacks the columns / joins.
    """
    parts = find_profit_parts(schema, column_types)
    if not parts:
        return None
    revenue_table, revenue_column = parts["revenue"]
    quantity_table, quantity_column = parts["quantity"]
    cost_table, cost_column = parts["cost"]

    # revenue and quantity must share one fact table so we can multiply cleanly
    if revenue_table != quantity_table:
        return None
    fact_table = revenue_table

    profit_expr = (
        f"SUM({fact_table}.{revenue_column} - "
        f"{fact_table}.{quantity_column} * {cost_table}.{cost_column})"
    )

    # column to group by (e.g. "region" -> store.region)
    dimension, dimension_table = find_dimension(
        query, schema, column_types, fact_table, revenue_column
    )

    # start FROM the fact table and JOIN in the other tables we reference
    from_clause = f"{database}.{fact_table}"
    needed_tables = []
    if cost_table != fact_table:
        needed_tables.append(cost_table)
    if dimension and dimension_table not in (fact_table, cost_table):
        needed_tables.append(dimension_table)
    for other_table in needed_tables:
        join_key = find_join_key(fact_table, other_table)
        if not join_key:
            return None  # cannot join safely -> let the normal path / LLM try
        from_clause += (
            f" JOIN {database}.{other_table} "
            f"ON {fact_table}.{join_key} = {other_table}.{join_key}"
        )

    # optional WHERE filter (keep the value's real stored casing)
    where_column, where_value, where_table = extract_where(query, schema)
    where_clause = ""
    if where_column and where_value:
        matched_value = resolve_where_value(where_table, where_column, where_value)
        safe_value = matched_value.replace("'", "''")
        where_clause = f" WHERE {where_column} = '{safe_value}'"

    # no dimension -> a single profit number
    if not dimension:
        return f"SELECT {profit_expr} AS profit FROM {from_clause}{where_clause}"

    dimension_sql = f"{dimension_table}.{dimension}"
    ascending = re.search(
        r"\b(lowest|smallest|least|bottom|ascending|asc|worst)\b", query
    )
    order = "ASC" if ascending else "DESC"
    sql = (
        f"SELECT {dimension_sql}, {profit_expr} AS profit FROM {from_clause}"
        f"{where_clause} GROUP BY {dimension_sql} ORDER BY profit {order}"
    )

    # "second highest" -> exactly the Nth row via LIMIT 1 OFFSET N-1
    nth = extract_nth_rank(query)
    if nth:
        return sql + f" LIMIT 1 OFFSET {nth - 1}"
    # a plain ranking ("top 5 ... by profit") still needs a LIMIT
    if intent == "Ranking":
        return sql + f" LIMIT {extract_limit(query)}"
    return sql


# ---------------------------------------------------------------------
# PART 6 : PUT IT ALL TOGETHER
# ---------------------------------------------------------------------

def generate_sql(query):
    """Turn a natural-language question into Hive SQL.

    Returns (sql_string, intent), or (None, intent) when the rules cannot
    build a query so the Ollama LLM can take over.

    "Top 5 customers with sales more than 5000" ->
        intent=Ranking, metric=SUM(sales_amount), dimension=customer_name,
        HAVING SUM(sales_amount) > 5000, JOIN on customer_id, LIMIT 5.
    """
    lower_query = query.lower()

    # 0. what kind of question is this?
    intent = detect_intent(query)

    # load everything we know about the dataset
    business_terms = load_business_terms()
    schema = load_schema()
    column_types = load_column_types()

    # SPECIAL CASE: real profit = revenue - quantity * cost. This needs an
    # inline expression (and usually a product join), which the generic metric
    # path cannot build, so we handle it here. Falls through to the normal
    # rules when the dataset does not have the needed columns / joins.
    if word_in_text("profit", lower_query) or word_in_text("margin", lower_query):
        profit_sql = build_profit_sql(
            lower_query, schema, column_types, intent, HIVE_DATABASE
        )
        if profit_sql:
            print("[RuleEngine] Derived profit metric (revenue - qty * cost)")
            return profit_sql, intent

    # 1. metric (the number to measure)
    term, metric_info = find_metric(lower_query, business_terms, schema, column_types)
    if not metric_info:
        print("[RuleEngine] No metric found; handing off to Ollama")
        return None, intent
    metric_table, metric_column, aggregation = metric_info

    # respect an explicit calculation in the wording (average/count/max/min)
    verb_aggregation = aggregation_from_verb(lower_query, intent)
    if verb_aggregation:
        print(f"[RuleEngine] Aggregation from wording: {verb_aggregation}")
        aggregation = verb_aggregation

    metric_sql = make_aggregation(aggregation, metric_column)
    alias = make_alias(term)

    database = HIVE_DATABASE

    # 2. dimension (the GROUP BY column)
    dimension, dimension_table = find_dimension(
        lower_query, schema, column_types, metric_table, metric_column
    )

    # 3. WHERE (original query so the value keeps its case)
    where_column, where_value, where_table = extract_where(query, schema)
    where_clause = ""
    if where_column and where_value:
        # resolve to the value's real stored casing ("east" -> "East",
        # "upi" -> "UPI"); falls back to Title Case if no match is found
        matched_value = resolve_where_value(where_table, where_column, where_value)
        safe_value = matched_value.replace("'", "''")
        where_clause = f" WHERE {where_column} = '{safe_value}'"

    # 4. HAVING (uses the final aggregation so it matches the SELECT)
    having_clause = extract_having(lower_query, metric_column, aggregation)

    # 4b. KPI "total revenue BY region" is really a grouped Aggregation. If a
    #     dimension is present and the wording asks per-group (or has HAVING),
    #     promote it so we emit a GROUP BY instead of one collapsed row.
    grouping_cue = any(
        word_in_text(cue, lower_query)
        for cue in ("by", "per", "across", "each", "grouped",
                    "distribution", "breakdown", "split", "segment")
    )
    if intent == "KPI" and dimension and (grouping_cue or having_clause):
        print("[RuleEngine] Promoting KPI -> Aggregation (dimension present)")
        intent = "Aggregation"

    # 5. JOIN if metric and dimension live in different tables
    from_clause = f"{database}.{metric_table}"
    if dimension and dimension_table != metric_table:
        join_sql, _, _ = find_join(metric_table, dimension_table)
        if join_sql:
            from_clause = join_sql
            print(f"[RuleEngine] Using JOIN: {from_clause}")

    # how to GROUP BY the dimension. A date/time dimension is bucketed into a
    # period so a trend gives one row per period, not per raw timestamp.
    group_by_expr = None
    group_select_expr = None
    if dimension:
        dimension_sql = f"{dimension_table}.{dimension}"  # qualify to avoid ambiguity
        if is_time_dimension(dimension_table, dimension, column_types):
            group_by_expr = make_time_bucket(dimension_sql, lower_query)
            group_select_expr = f"{group_by_expr} AS period"
        else:
            group_by_expr = dimension_sql
            group_select_expr = dimension_sql

    # 6. build the final SQL, shaped by the intent
    return build_sql_for_intent(
        intent=intent,
        query=lower_query,
        original_query=query,
        metric_sql=metric_sql,
        alias=alias,
        from_clause=from_clause,
        group_by_expr=group_by_expr,
        group_select_expr=group_select_expr,
        where_clause=where_clause,
        having_clause=having_clause,
    )


def build_order_by(query, alias, group_by_expr, where_clause):
    """Decide the ORDER BY for a grouped result (smarter than always DESC).

    - If the grouped column is pinned by an equality filter (e.g. "revenue
      where region is west" grouped by region) the result is a single row,
      so we skip ORDER BY entirely - sorting one row is meaningless.
    - Otherwise sort by the metric: biggest first by default, but follow the
      wording when the user asks for ascending / lowest-first.
    """
    # single-group case: WHERE <col> = ... together with GROUP BY that <col>
    if where_clause and group_by_expr:
        filter_match = re.search(r"WHERE\s+([\w.]+)\s*=", where_clause)
        if filter_match:
            filtered_column = filter_match.group(1).split(".")[-1].lower()
            grouped_column = group_by_expr.split(".")[-1].lower()
            if filtered_column == grouped_column:
                return ""

    ascending = re.search(
        r"\b(ascending|asc|increasing|lowest|smallest|least|bottom)\b", query
    )
    direction = "ASC" if ascending else "DESC"
    return f" ORDER BY {alias} {direction}"


def build_sql_for_intent(intent, query, original_query, metric_sql, alias,
                         from_clause, group_by_expr, group_select_expr,
                         where_clause, having_clause):
    """Assemble the SELECT in the right shape for the intent.

    group_by_expr = expression to GROUP BY / ORDER BY (dimension or date
    bucket); group_select_expr = what to show for it. Both None if no dim.
    """

    # KPI -> one single number, no grouping
    if intent == "KPI":
        sql = f"SELECT {metric_sql} AS {alias} FROM {from_clause}{where_clause}"
        return sql, intent

    # Ranking -> top / bottom N rows, needs something to rank by
    if intent == "Ranking":
        if not group_by_expr:
            print("[RuleEngine] Ranking without a dimension; handing off to Ollama")
            return None, intent
        descending = not any(word in query for word in ["bottom", "lowest", "worst"])
        order = "DESC" if descending else "ASC"
        # "second highest" / "3rd largest" -> exactly the Nth row via
        # LIMIT 1 OFFSET N-1 (no sub-query, no window function).
        nth = extract_nth_rank(query)
        if nth:
            limit_clause = f" LIMIT 1 OFFSET {nth - 1}"
        else:
            limit_clause = f" LIMIT {extract_limit(query)}"
        sql = (
            f"SELECT {group_select_expr}, {metric_sql} AS {alias} FROM {from_clause}"
            f"{where_clause} GROUP BY {group_by_expr}{having_clause}"
            f" ORDER BY {alias} {order}{limit_clause}"
        )
        return sql, intent

    # Trend -> a value over time, ordered by the time period
    if intent == "Trend":
        if not group_by_expr:
            return None, intent
        sql = (
            f"SELECT {group_select_expr}, {metric_sql} AS {alias} FROM {from_clause}"
            f"{where_clause} GROUP BY {group_by_expr}{having_clause}"
            f" ORDER BY {group_by_expr} ASC"
        )
        return sql, intent

    # Comparison -> if the user named two things, filter to just those two
    if intent == "Comparison" and group_by_expr and not where_clause:
        values = extract_comparison_values(original_query)
        if len(values) == 2:
            value_list = ", ".join(f"'{value}'" for value in values)
            where_clause = f" WHERE {group_by_expr} IN ({value_list})"

    # Aggregation / Comparison / Visualization -> group by the dimension
    if group_by_expr:
        order_by = build_order_by(query, alias, group_by_expr, where_clause)
        sql = (
            f"SELECT {group_select_expr}, {metric_sql} AS {alias} FROM {from_clause}"
            f"{where_clause} GROUP BY {group_by_expr}{having_clause}"
            f"{order_by}"
        )
        return sql, intent

    # No dimension but a HAVING needs a GROUP BY -> hand off to Ollama.
    if having_clause:
        print("[RuleEngine] HAVING without a dimension; handing off to Ollama")
        return None, intent

    # Otherwise just the filtered single number.
    sql = f"SELECT {metric_sql} AS {alias} FROM {from_clause}{where_clause}"
    return sql, intent


# ---------------------------------------------------------------------
# QUICK TEST : python rule_engine.py (needs the MySQL metadata populated)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What is the total revenue",
        "Show top 5 customers by revenue",
        "Show 8 customers by revenue",
        "List 3 cities by revenue",
        "Revenue by city",
        "Total profit",
        "How much did we sell in total",
        "Best sellers in our catalog",
        "Earnings distributed across branches",
        "Who are our worst performers",
        "Revenue where region = 'West'",
        "Customers with revenue more than 5000",
        "Regions where total sales exceed 10000",
        "Top 5 customers with sales more than 5000",
        "Compare sales between Pune and Mumbai",
        "Give me second highest profit by region",
        "Second highest revenue by region",
    ]
    for test_query in test_queries:
        sql, intent = generate_sql(test_query)
        print(f"\nQ: {test_query}\n  intent = {intent}\n  sql = {sql}")

