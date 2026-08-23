import os
import json
from datetime import datetime
from collections import defaultdict

from nlp.intent import get_intent_confidence
from query_engine.rule_engine import generate_sql

# Keep the dataset and the report next to this file, so the script works
# no matter which folder we run it from.
HERE = os.path.dirname(__file__)
EVAL_FILE = os.path.join(HERE, "eval_dataset.json")
REPORT_FILE = os.path.join(HERE, "accuracy_report.txt")

# The order in which difficulty levels should appear in the report.
DIFFICULTY_ORDER = ["easy", "medium", "hard", "edge"]


def load_dataset():
 
    with open(EVAL_FILE) as f:
        return json.load(f)



def check_intent(case):
   
    predicted_intent, confidence = get_intent_confidence(case["query"])
    is_correct = predicted_intent == case["expected_intent"]
    return predicted_intent, confidence, is_correct


def check_sql(case):

    # 1. Forecasting queries are handled elsewhere, so skip SQL here.
    if case.get("skip_sql"):
        return "(skipped - forecasting)", True

    # 2. Some queries SHOULD fail (gibberish, greetings, ...).
    if case.get("expect_fail"):
        sql, _ = generate_sql(case["query"])
        return (sql or "(None - expected)"), sql is None

    # 3. Normal queries: generate SQL and look for the expected keywords.
    sql, _ = generate_sql(case["query"])
    sql = sql or ""
    expected_keywords = case.get("expect_sql_contains", [])
    if expected_keywords:
        is_correct = all(word.upper() in sql.upper() for word in expected_keywords)
    else:
        is_correct = len(sql) > 0
    return sql, is_correct


def percent(part, whole):
    
    return round(part / whole * 100, 2) if whole else 0.0


def run_all_cases(dataset):

    intent_correct = 0
    sql_correct = 0
    details = []
    confusion = defaultdict(lambda: defaultdict(int))
    diff_scores = defaultdict(lambda: {"intent_ok": 0, "sql_ok": 0, "total": 0})

    for case in dataset:
        query = case["query"]
        difficulty = case.get("difficulty", "easy")
        diff_scores[difficulty]["total"] += 1

        # ---- intent accuracy ----
        predicted_intent, confidence, intent_ok = check_intent(case)
        intent_correct += int(intent_ok)
        confusion[case["expected_intent"]][predicted_intent] += 1

        # ---- SQL correctness ----
        sql, sql_ok = check_sql(case)
        sql_correct += int(sql_ok)

        # ---- per-difficulty tallies ----
        diff_scores[difficulty]["intent_ok"] += int(intent_ok)
        diff_scores[difficulty]["sql_ok"] += int(sql_ok)

        details.append((query, case["expected_intent"], predicted_intent,
                        confidence, intent_ok, sql_ok, sql, difficulty))

    return {
        "intent_correct": intent_correct,
        "sql_correct": sql_correct,
        "confusion": confusion,
        "diff_scores": diff_scores,
        "details": details,
    }


def build_header(total, intent_acc, sql_acc, overall):
    
    return [
        "=" * 70,
        "TARK AI - ACCURACY EVALUATION REPORT",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 70,
        f"Test cases: {total}",
        f"Intent Detection Accuracy: {intent_acc}%",
        f"SQL Correctness: {sql_acc}%",
        f"Overall System Accuracy: {overall}%",
        "",
        "-" * 70,
        "PER-DIFFICULTY BREAKDOWN",
        "-" * 70,
    ]


def build_difficulty_section(diff_scores):
    
    lines = []
    for difficulty in DIFFICULTY_ORDER:
        scores = diff_scores[difficulty]
        if scores["total"] > 0:
            intent_pct = round(scores["intent_ok"] / scores["total"] * 100, 1)
            sql_pct = round(scores["sql_ok"] / scores["total"] * 100, 1)
            lines.append(f"  {difficulty.upper():8s}: {scores['total']:3d} cases | "
                         f"Intent: {intent_pct}% | SQL: {sql_pct}%")
    return lines


def collect_all_intents(confusion):
    
    expected_labels = list(confusion.keys())
    predicted_labels = [i for row in confusion.values() for i in row.keys()]
    return sorted(set(expected_labels + predicted_labels))


def build_confusion_section(confusion, all_intents):
    
    lines = ["", "-" * 70, "INTENT CONFUSION MATRIX", "-" * 70]
    lines.append(f"{'Expected':<18}" + "".join(f"{i:<14}" for i in all_intents))
    for expected in all_intents:
        row = f"{expected:<18}" + "".join(f"{confusion[expected][p]:<14}" for p in all_intents)
        lines.append(row)
    return lines


def build_per_intent_section(confusion, all_intents):
    
    lines = ["", "-" * 70, "PER-INTENT ACCURACY", "-" * 70]
    for intent in all_intents:
        total_for_intent = sum(confusion[intent].values())
        correct_for_intent = confusion[intent].get(intent, 0)
        if total_for_intent > 0:
            acc = round(correct_for_intent / total_for_intent * 100, 1)
            lines.append(f"  {intent:<18}: {correct_for_intent}/{total_for_intent} = {acc}%")
    return lines


def build_details_section(details):
    
    lines = ["", "-" * 70, "DETAILED RESULTS", "-" * 70]
    for query, expected, predicted, conf, intent_ok, sql_ok, sql, difficulty in details:
        intent_mark = "OK" if intent_ok else "X "
        sql_mark = "OK" if sql_ok else "X "
        lines.append(f"[{difficulty:6s}] Q: {query}")
        lines.append(f"         intent expected={expected} predicted={predicted} "
                     f"conf={conf}% [{intent_mark}]")
        lines.append(f"         sql [{sql_mark}]: {sql[:100]}")
    return lines


def build_report(results, total, intent_acc, sql_acc, overall):
    
    confusion = results["confusion"]
    all_intents = collect_all_intents(confusion)

    lines = []
    lines += build_header(total, intent_acc, sql_acc, overall)
    lines += build_difficulty_section(results["diff_scores"])
    lines += build_confusion_section(confusion, all_intents)
    lines += build_per_intent_section(confusion, all_intents)
    lines += build_details_section(results["details"])
    return "\n".join(lines)


def save_report(report):
   
    with open(REPORT_FILE, "w") as f:
        f.write(report)


def evaluate():
    
    dataset = load_dataset()
    total = len(dataset)

    print(f"[Eval] Running evaluation on {total} test cases...")
    results = run_all_cases(dataset)

    # Turn the running totals into percentages.
    intent_acc = percent(results["intent_correct"], total)
    sql_acc = percent(results["sql_correct"], total)
    overall = round((intent_acc + sql_acc) / 2, 2)

    print(f"[Eval] Intent Accuracy: {intent_acc}%")
    print(f"[Eval] SQL Correctness: {sql_acc}%")
    print(f"[Eval] Overall: {overall}%")

    # Build, save, and show the detailed report.
    report = build_report(results, total, intent_acc, sql_acc, overall)
    save_report(report)
    print(report)
    print(f"\nReport saved to {REPORT_FILE}")

    return {
        "intent_accuracy": intent_acc,
        "sql_accuracy": sql_acc,
        "overall": overall,
        "confusion_matrix": dict(results["confusion"]),
        "difficulty_scores": dict(results["diff_scores"]),
        "total_cases": total,
    }


if __name__ == "__main__":
    evaluate()
