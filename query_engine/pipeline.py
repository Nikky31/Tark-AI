# query_engine/pipeline.py
# =====================================================================
# THE FULL QUERY PIPELINE
# ---------------------------------------------------------------------
# This file ties all the layers together and is the single entry point
# that the Streamlit app calls. Give it an English question and it walks
# the question through every stage of Tark AI:
#
#   Layer 6  Intent detection  -> what KIND of question is this?
#   Layer 7  Validation        -> is this a safe / answerable question?
#   Layer 8  Rule-based SQL     -> try to build SQL with our own rules
#   Layer 8b Ollama fallback    -> if the rules cannot, ask the LLM
#   Layer 9  Execution          -> run the SQL on Hive (and log it)
#
# The whole journey is packaged up into one result dictionary that the
# app can read from.
# =====================================================================

from nlp.intent import detect_intent
from nlp.validate import validate_query
from query_engine.rule_engine import generate_sql
from query_engine.fallback import generate_sql_ollama
from query_engine.executor import execute_query


def make_empty_result(query):
    """Create the result dictionary we fill in as the query moves along.

    Keeping every possible key here (even if it stays empty) means the app
    always gets the same shape back, no matter where the pipeline stops.

    Keys:
        query   -> the original question the user asked
        intent  -> the detected intent (e.g. "Ranking")
        valid   -> did the question pass validation?
        message -> a human-readable note (error / status)
        sql     -> the SQL we generated
        engine  -> who built the SQL: "rule" or "ollama"
        result  -> the execution result (rows, timing, etc.)
    """
    return {
        "query": query,
        "intent": None,
        "valid": False,
        "message": "",
        "sql": None,
        "engine": None,
        "result": None,
    }


def build_sql(query):
    """Turn the question into SQL, trying our rules first, then the LLM.

    Returns a (sql, engine, intent) tuple:
      - First we ask the rule engine (Layer 8). If it succeeds we use that
        SQL and mark the engine as "rule".
      - If the rule engine returns nothing, we fall back to the Ollama LLM
        (Layer 8b) and mark the engine as "ollama".

    The rule engine also returns the intent it detected, so we hand that
    back too, matching the original behavior.
    """
    sql, intent = generate_sql(query)

    if sql:
        return sql, "rule", intent

    # Rules could not handle it -> let the LLM try.
    sql = generate_sql_ollama(query)
    return sql, "ollama", intent


def process(query, run=True):
    """Run a question through the whole pipeline, end to end.

    Args:
        query: the user's natural-language question.
        run:   if True, also execute the SQL on Hive. Set to False when you
               only want to SEE the generated SQL without running it.

    Returns:
        The result dictionary described in make_empty_result().
    """
    result = make_empty_result(query)

    # ---- Layer 6 : intent detection ----
    # Work out what kind of question this is (KPI, Ranking, Trend, ...).
    result["intent"] = detect_intent(query)

    # ---- Layer 7 : validation ----
    # Stop early if the question is not safe / answerable.
    is_valid, message = validate_query(query)
    result["valid"] = is_valid
    result["message"] = message
    if not is_valid:
        return result

    # ---- Layer 8 / 8b : build the SQL ----
    # Try the rule engine first, then fall back to the Ollama LLM.
    sql, engine, intent = build_sql(query)
    result["intent"] = intent
    result["sql"] = sql
    result["engine"] = engine

    # If neither approach produced SQL, tell the user and stop.
    if not result["sql"]:
        result["message"] = "Could not generate SQL for this query."
        return result

    # ---- Layer 9 : execution + logging ----
    # Only run the SQL when asked to.
    if run:
        result["result"] = execute_query(
            result["sql"],
            user_query=query,
            engine=result["engine"],
            intent=result["intent"],
        )

    return result


# =====================================================================
# QUICK TEST
# =====================================================================
# Run this file directly (python pipeline.py) to try one example query
# through the full pipeline and print what came back.

if __name__ == "__main__":
    example_query = "Show top 5 customers by revenue"
    result = process(example_query)

    print(f"\nQ: {example_query}")
    print("Intent:", result["intent"], "| Engine:", result["engine"])
    print("SQL:", result["sql"])

    if result["result"] and result["result"]["success"]:
        print("Rows:", result["result"]["row_count"],
              "| Time:", result["result"]["execution_time"])
        print(result["result"]["dataframe"])
