# analytics/money.py
# ---------------------------------------------------------------------
# CURRENCY FORMATTING (shared helper)
#
# The Ask Analytics page shows money columns (revenue, profit, loss ...)
# in three places: the result table, the auto chart and the business
# insights. This small module keeps the "how do we show money" rules in
# ONE place so all three always look the same.
#
# What the user asked for:
#   - put a currency symbol in front (default "$" when none is known)
#   - shorten big numbers -> 1.2K (thousands), 3.4M (millions), 5.6B ...
# ---------------------------------------------------------------------

import pandas as pd

# symbol to use when the column / dataset does not show one
DEFAULT_SYMBOL = "$"

# words in a column name that tell us the column holds money
CURRENCY_HINTS = [
    "revenue", "profit", "loss", "sales", "amount", "cost", "price",
    "income", "earning", "margin", "turnover", "spend", "expense",
    "salary", "budget", "payment", "discount", "fee", "balance",
]

# currency symbols / codes we can recognise inside a column name
KNOWN_SYMBOLS = ["$", "\u20b9", "\u20ac", "\u00a3", "\u00a5"]
CODE_TO_SYMBOL = {
    "usd": "$", "inr": "\u20b9", "rs": "\u20b9", "rupee": "\u20b9",
    "eur": "\u20ac", "gbp": "\u00a3", "jpy": "\u00a5",
}


def is_currency_column(name):
    """True when a column name looks like money (revenue, profit, amount ...)."""
    low = str(name).lower()
    return any(word in low for word in CURRENCY_HINTS)


def detect_symbol(name):
    """Find a currency symbol / code inside the column name, else return "$".

    Lets a column like "revenue_inr" or "amount_rupee" show the right symbol,
    while a plain "profit" column falls back to the default "$".
    """
    text = str(name)
    for sym in KNOWN_SYMBOLS:
        if sym in text:
            return sym
    low = text.lower()
    for code, sym in CODE_TO_SYMBOL.items():
        if code in low:
            return sym
    return DEFAULT_SYMBOL


def abbreviate_number(value):
    """Shorten a number to a K / M / B / T string (1172123.4 -> "1.17M").

    Keeps up to two decimals and drops any that are only zeros. Values under
    a thousand are shown as-is. Non-numbers (NaN, text) are returned unchanged.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "-" if num < 0 else ""
    num = abs(num)

    for suffix, size in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if num >= size:
            short = f"{num / size:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{short}{suffix}"

    # small value: whole numbers stay whole, others keep up to 2 decimals
    if num == int(num):
        return f"{sign}{int(num)}"
    return f"{sign}{num:.2f}".rstrip("0").rstrip(".")


def format_money(value, symbol=DEFAULT_SYMBOL):
    """Format one money value as "$1.17M" (symbol + short number)."""
    return f"{symbol}{abbreviate_number(value)}"


def format_value(column, value):
    """Format a value for display based on its column name.

    Money columns become "$1.2M"; other numbers keep plain thousands
    separators (1,234); anything non-numeric is returned as text.
    """
    if is_currency_column(column):
        return format_money(value, detect_symbol(column))
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num == int(num):
        return f"{int(num):,}"
    return f"{num:,.2f}"


def format_dataframe(df):
    """Return a copy of the table with money columns shown as "$1.2M".

    Only numeric currency columns are reformatted (into text); every other
    column is left exactly as it was so counts, ids and labels do not change.
    """
    display = df.copy()
    for col in display.columns:
        if is_currency_column(col) and pd.api.types.is_numeric_dtype(df[col]):
            symbol = detect_symbol(col)
            display[col] = df[col].apply(lambda v: format_money(v, symbol))
    return display
