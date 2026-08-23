import pandas as pd
from analytics.money import format_value


def metric_insights(df, dim, metric):
    total = df[metric].sum()
    top = df.loc[df[metric].idxmax()]
    bottom = df.loc[df[metric].idxmin()]

    insights = [
        f"Highest {metric}: {top[dim]} ({format_value(metric, top[metric])}).",
        f"Lowest {metric}: {bottom[dim]} ({format_value(metric, bottom[metric])}).",
        f"Total {metric} across all {dim}: {format_value(metric, total)}.",
    ]
    if total > 0:
        share = round((top[metric] / total) * 100, 1)
        insights.append(f"{top[dim]} contributes {share}% of total {metric}.")

    return insights


def trend_insight(df, metric):
    if len(df) <= 1:
        return None
    first, last = df[metric].iloc[0], df[metric].iloc[-1]
    if first == 0:
        return None
    change = round(((last - first) / first) * 100, 1)
    direction = "increased" if change >= 0 else "decreased"
    return f"{metric} {direction} by {abs(change)}% over the period."


def generate_insights(df, intent="Aggregation"):
    if df is None or df.empty:
        return ["No data available for insights."]

    cols = list(df.columns)

    if len(cols) == 1 and len(df) == 1:
        return [f"{cols[0]} = {format_value(cols[0], df.iloc[0, 0])}"]

    if len(cols) < 2:
        return [f"Returned {len(df)} rows."]

    dim, metric = cols[0], cols[1]

    if not pd.api.types.is_numeric_dtype(df[metric]):
        return [f"Returned {len(df)} rows across {df[dim].nunique()} {dim} values."]

    insights = metric_insights(df, dim, metric)

    if intent == "Trend":
        trend = trend_insight(df, metric)
        if trend:
            insights.append(trend)

    return insights

