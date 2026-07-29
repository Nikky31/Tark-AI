# nlp/validate.py
# =====================================================================
# LAYER 7 : QUERY VALIDATION
# ---------------------------------------------------------------------
# What this file does (in plain English):
#
# Before we spend effort building SQL, we do a quick sanity check on the
# user's question. The idea is simple:
#
#   "Does this question mention at least ONE thing we actually know about
#    the loaded dataset?" (a table, a column, or a business term)
#
#   * If YES -> the query is valid, carry on to SQL generation.
#   * If NO  -> maybe the user just made a typo, so we use difflib to
#              suggest the closest known word ("revanue" -> "revenue").
#              Otherwise we politely say we could not understand it.
#
# The function returns a simple pair:  (is_valid, message)
#
# IMPORTANT: we build our "known words" list from the SAME vocabulary the
# rule engine uses (schema + business terms + English synonyms + dimension
# hints). That way, any query the rule engine could actually answer also
# passes validation here.
# =====================================================================

import re
import difflib

from config.db import fetch_all

# We reuse the rule engine's word lists so validation and SQL generation
# agree on what counts as a "known" word. If that import ever fails we
# fall back to empty lists, so validation still works using just the
# schema and business terms.
#
# (The dimension-hints constant is called DIMENSION_HINTS in newer code
#  and DIM_HINTS in older code, so we try both names.)
try:
    from query_engine.rule_engine import ENGLISH_SYNONYMS
    try:
        from query_engine.rule_engine import DIMENSION_HINTS
    except ImportError:
        from query_engine.rule_engine import DIM_HINTS as DIMENSION_HINTS
except Exception as error:  # pragma: no cover
    print(f"[Validate] Could not import rule engine vocab ({error}); "
          f"falling back to schema/business terms only.")
    ENGLISH_SYNONYMS, DIMENSION_HINTS = {}, {}


# Everyday filler words that carry no data meaning. We never suggest a
# typo correction for these (nobody misspells "revenue" as "please").
STOP_WORDS = {
    "show", "give", "list", "find", "display", "tell", "please", "about",
    "between", "across", "over", "with", "from", "that", "this", "have",
    "want", "need", "would", "could", "there", "their", "where", "which",
    "what", "whats", "were", "does", "much", "many",
}


# ---------------------------------------------------------------------
# STEP 1 : LOAD WHAT WE KNOW FROM MYSQL
# ---------------------------------------------------------------------

def load_schema():
    """Read every table and column from the schema_context table.

    Returns two things:
      schema      -> a dict of {table_name: [list of its columns]}
      all_columns -> a flat list of every column name
    """
    rows = fetch_all("SELECT table_name, column_name FROM schema_context")
    schema = {}
    all_columns = []
    for table_name, column_name in rows:
        schema.setdefault(table_name, []).append(column_name)
        all_columns.append(column_name)
    return schema, all_columns


def load_business_terms():
    """Read the list of business terms (e.g. "revenue", "profit")."""
    rows = fetch_all("SELECT DISTINCT business_term FROM business_context")
    return [row[0] for row in rows]


# ---------------------------------------------------------------------
# STEP 2 : BUILD ONE BIG LIST OF "KNOWN" WORDS
# ---------------------------------------------------------------------

def build_vocabulary(schema, all_columns, business_terms):
    """Collect every word we consider "known" into one set.

    This is the master list a query is checked against. It includes:
      * table names
      * full column names AND their word-parts
        (so "total_amount" also adds "total" and "amount")
      * business terms
      * every key and value from ENGLISH_SYNONYMS
      * every key and value from DIMENSION_HINTS

    We use a set so duplicates are removed automatically. At the end we
    drop empty strings, pure numbers, and the bare "id"/"key" tokens
    because those are not useful things for a user to mention.
    """
    vocabulary = set()

    # table names
    for table_name in schema:
        vocabulary.add(table_name.lower())

    # column names + their individual word-parts
    for column_name in all_columns:
        column_lower = column_name.lower()
        vocabulary.add(column_lower)
        for part in column_lower.replace("_", " ").split():
            if len(part) >= 3:
                vocabulary.add(part)

    # business terms
    for term in business_terms:
        vocabulary.add(term.lower())

    # English synonym keys and values
    for key, synonyms in ENGLISH_SYNONYMS.items():
        vocabulary.add(key.lower())
        for synonym in synonyms:
            vocabulary.add(synonym.lower())

    # dimension-hint keys and values
    for key, hints in DIMENSION_HINTS.items():
        vocabulary.add(key.lower())
        for hint in hints:
            vocabulary.add(hint.lower())

    # remove blanks, plain numbers, and bare id/key tokens
    return {
        word for word in vocabulary
        if word and not word.isdigit() and word not in {"id", "key"}
    }


# ---------------------------------------------------------------------
# STEP 3 : LOOK AT THE USER'S WORDS
# ---------------------------------------------------------------------

def get_query_word_forms(query):
    """Split the query into words, plus simple singular/plural variants.

    We add both forms of each word so that "customers" can match the known
    word "customer" and "customer" can match "customers".

    Returns (tokens, forms):
      tokens -> the exact words found in the query
      forms  -> those words plus their singular/plural variants
    """
    tokens = set(re.findall(r"[a-z0-9_]+", query))
    forms = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            forms.add(token[:-1])   # customers -> customer
        elif len(token) >= 3:
            forms.add(token + "s")  # customer -> customers
    return tokens, forms


def find_referenced_terms(query, vocabulary):
    """Return every known term the query mentions (sorted).

    Two kinds of terms are handled:
      * multi-word terms like "total amount" -> matched as a phrase
        (must appear as-is somewhere in the query)
      * single-word terms -> matched against the query's word forms
        (so plural/singular still counts as a match)
    """
    _, word_forms = get_query_word_forms(query)
    found = set()
    for term in vocabulary:
        if " " in term:          # phrase match for multi-word terms
            if term in query:
                found.add(term)
        elif term in word_forms:  # single-word match (with plural handling)
            found.add(term)
    return sorted(found)


def suggest_correction(word, known_words):
    """Return the closest known word to `word`, or None if nothing is close.

    This is our typo helper. difflib compares the misspelled word to every
    known word and returns the best match, but only if it is at least 60%
    similar (cutoff=0.6), so we do not suggest wild guesses.
    """
    matches = difflib.get_close_matches(word, known_words, n=1, cutoff=0.6)
    return matches[0] if matches else None


# ---------------------------------------------------------------------
# STEP 4 : THE MAIN VALIDATION FUNCTION
# ---------------------------------------------------------------------

def validate_query(query):
    """Check whether a query mentions anything we know about.

    Returns (is_valid, message):
      * (True,  "Valid. Referenced: [...]")  if it mentions a known term
      * (False, "...Did you mean 'x'?")      if we can suggest a typo fix
      * (False, "...")                        otherwise
    """
    schema, all_columns = load_schema()
    business_terms = load_business_terms()

    # if no dataset has been loaded yet, there is nothing to validate against
    if not schema:
        return False, "No tables found. Please upload a dataset first."

    lower_query = query.lower()
    vocabulary = build_vocabulary(schema, all_columns, business_terms)

    # the query is valid if it references at least one known term
    referenced = find_referenced_terms(lower_query, vocabulary)
    if referenced:
        return True, f"Valid. Referenced: {referenced}"

    # nothing matched -> try to suggest a typo correction. We check the
    # longest meaningful words first, since those are the most likely to
    # be the important (misspelled) keyword.
    candidate_words = [
        word for word in re.findall(r"[a-z]+", lower_query)
        if len(word) > 3 and word not in STOP_WORDS
    ]
    candidate_words.sort(key=len, reverse=True)

    for word in candidate_words:
        suggestion = suggest_correction(word, list(vocabulary))
        if suggestion:
            return False, f"Could not match '{word}'. Did you mean '{suggestion}'?"

    # no known terms and no good suggestion -> give up gracefully
    return False, "Query does not reference any known table, column, or KPI."


# ---------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------
# Run this file directly (python validate.py) to try a few examples.

if __name__ == "__main__":
    test_queries = [
        "total revenue",
        "show revanue",
        "top customers by sales",
        "earnings by city",
        "xyz abc",
    ]
    for test_query in test_queries:
        print(test_query, "->", validate_query(test_query))
