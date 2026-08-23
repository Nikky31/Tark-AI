
import pandas as pd

DEFAULT_SYMBOL = "$"

CURRENCY_HINTS = [
    "revenue", "profit", "loss", "sales", "amount", "cost", "price",
    "income", "earning", "margin", "turnover", "spend", "expense",
    "salary", "budget", "payment", "discount", "fee", "balance",
]

KNOWN_SYMBOLS = ["$", "\u20b9", "\u20ac", "\u00a3", "\u00a5"]
CODE_TO_SYMBOL = {
    "usd": "$", "inr": "\u20b9", "rs": "\u20b9", "rupee": "\u20b9",
    "eur": "\u20ac", "gbp": "\u00a3", "jpy": "\u00a5",
}


def is_currency_column(name):
    low = str(name).lower()
    return any(word in low for word in CURRENCY_HINTS)


def detect_symbol(name):
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
        
    if num == int(num):
        return f"{sign}{int(num)}"
    return f"{sign}{num:.2f}".rstrip("0").rstrip(".")


def format_money(value, symbol=DEFAULT_SYMBOL):
    return f"{symbol}{abbreviate_number(value)}"


def format_value(column, value):
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
    display = df.copy()
    for col in display.columns:
        if is_currency_column(col) and pd.api.types.is_numeric_dtype(df[col]):
            symbol = detect_symbol(col)
            display[col] = df[col].apply(lambda v: format_money(v, symbol))
    return display
