# analytics/insights.py
# ---------------------------------------------------------------------
# LAYER 12 : BUSINESS INSIGHTS (rule-based)
#
# After a query runs we have a small result table (a pandas DataFrame).
# This layer reads that table and writes short, plain-English sentences
# that explain what the numbers mean (highest, lowest, total, share and
# the trend direction).
#
# It is 100% rule-based pandas logic - no external API or AI is used, so
# it is fast and always gives the same answer for the same data.
# ---------------------------------------------------------------------

import pandas as pd

from analytics.money import format_value


def metric_insights(df, dim, metric):
    """Build the highest / lowest / total / share sentences for a metric.

    Given a category column (dim) and a numeric column (metric) this finds:
      - which category has the biggest value
      - which category has the smallest value
      - the grand total of the metric
      - how big a share the top category is (only when the total is > 0)

    Returns the list of insight strings in that order.
    """
    total = df[metric].sum()
    top = df.loc[df[metric].idxmax()]
    bottom = df.loc[df[metric].idxmin()]

    insights = [
        f"Highest {metric}: {top[dim]} ({format_value(metric, top[metric])}).",
        f"Lowest {metric}: {bottom[dim]} ({format_value(metric, bottom[metric])}).",
        f"Total {metric} across all {dim}: {format_value(metric, total)}.",
    ]

    # contribution of the top item (skip if total is 0 to avoid /0)
    if total > 0:
        share = round((top[metric] / total) * 100, 1)
        insights.append(f"{top[dim]} contributes {share}% of total {metric}.")

    return insights


def trend_insight(df, metric):
    """Describe whether the metric went up or down over the period.

    Compares the first and last values of a time-ordered result. Returns a
    single sentence, or None when there is nothing meaningful to say (only
    one row, or the first value is 0 so a percentage cannot be computed).
    """
    if len(df) <= 1:
        return None
    first, last = df[metric].iloc[0], df[metric].iloc[-1]
    if first == 0:
        return None
    change = round(((last - first) / first) * 100, 1)
    direction = "increased" if change >= 0 else "decreased"
    return f"{metric} {direction} by {abs(change)}% over the period."


def generate_insights(df, intent="Aggregation"):
    """Return a list of plain-language insight strings for a result table.

    Handles the simple cases first (no data, a single KPI number, or a
    non-numeric result), then builds the full metric insights and adds a
    trend sentence for time-based ("Trend") queries.
    """
    if df is None or df.empty:
        return ["No data available for insights."]

    cols = list(df.columns)

    # single KPI value (one column, one row)
    if len(cols) == 1 and len(df) == 1:
        return [f"{cols[0]} = {format_value(cols[0], df.iloc[0, 0])}"]

    if len(cols) < 2:
        return [f"Returned {len(df)} rows."]

    dim, metric = cols[0], cols[1]

    # if the value column is not numeric we can only report the shape
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return [f"Returned {len(df)} rows across {df[dim].nunique()} {dim} values."]

    insights = metric_insights(df, dim, metric)

    # trend direction only makes sense for time-based queries
    if intent == "Trend":
        trend = trend_insight(df, metric)
        if trend:
            insights.append(trend)

    return insights

