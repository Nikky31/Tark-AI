# analytics/charts.py
# ---------------------------------------------------------------------
# LAYER 11 : VISUALIZATION
#
# After the SQL runs we get a small table (a pandas DataFrame). This file
# looks at that table AND the detected intent (KPI, Ranking, Trend, ...)
# and draws a colourful Plotly chart for it.
#
# Two simple steps keep it readable:
#   1. pick_chart_type() -> decides WHICH chart suits the data
#   2. make_*() builders -> actually BUILD that one chart (one each)
#
# build_chart() glues the two steps together. app.py only calls
# build_chart() and build_forecast_chart(), so those two names must stay.
# ---------------------------------------------------------------------

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.money import is_currency_column, detect_symbol


# ---------------------------------------------------------------------
# COLOURS
# ---------------------------------------------------------------------

# a bright, friendly palette; Plotly picks these one-by-one for bars/slices
CHART_COLORS = [
    "#2E86DE",  # blue
    "#EE5253",  # red
    "#10AC84",  # green
    "#F79F1F",  # orange
    "#5F27CD",  # purple
    "#00D2D3",  # teal
    "#FF9FF3",  # pink
    "#1DD1A1",  # mint
    "#FECA57",  # yellow
    "#576574",  # grey
]

# text / accent colour for titles and axis labels
TEXT_COLOR = "#2C3E50"


def style_chart(fig, height=460):
    """Give every chart the same clean, consistent look (written once, reused).

    Sets a light transparent 'plotly_white' template, a readable title/axis
    font, comfortable margins and a fixed height. Returns the same figure
    so calls can be chained.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, color=TEXT_COLOR),
        font=dict(family="Arial", size=13, color=TEXT_COLOR),
        margin=dict(t=60, l=50, r=30, b=50),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------
# STEP 1 : decide which chart to draw
# ---------------------------------------------------------------------

def pick_chart_type(intent, data):
    """Pick the chart type that best fits the intent and the data.

    Different questions read best as different charts (Trend -> area,
    Ranking -> horizontal bar, etc.). If there is no data or fewer than two
    columns there is nothing to plot, so we return None.

    Returns a short keyword ("bar_v", "bar_h", "area", "donut", "treemap",
    "kpi") or None.
    """
    if data is None or data.empty or len(data.columns) < 2:
        return None

    # how many different labels are in the first (category) column
    first_column = data.columns[0]
    unique_labels = data[first_column].nunique()

    if intent == "Trend":
        return "area"

    if intent == "Ranking":
        return "bar_h"

    if intent == "Comparison":
        return "bar_v"

    if intent == "Aggregation":
        # few categories look great as a donut, many are clearer as a treemap
        if unique_labels <= 5:
            return "donut"
        if unique_labels > 12:
            return "treemap"
        return "bar_v"

    if intent == "Visualization":
        if unique_labels > 8:
            return "treemap"
        return "bar_v"

    if intent == "KPI":
        # a single number is shown by app.py directly, not as a chart
        return "kpi"

    # anything else -> a simple vertical bar is a safe default
    return "bar_v"


# ---------------------------------------------------------------------
# STEP 2 : the individual chart builders (one function per chart)
# ---------------------------------------------------------------------

def make_vertical_bar(data, category_col, value_col):
    """Colourful vertical bar chart: compare a value across a few categories.

    Each bar gets its own colour and the value is printed on top.
    """
    fig = px.bar(
        data,
        x=category_col,
        y=value_col,
        color=category_col,               # colour each bar differently
        text=value_col,                   # show the number on the bar
        title=f"{value_col} by {category_col}",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(showlegend=False)   # colours are obvious, no legend needed
    return style_chart(fig)


def make_horizontal_bar(data, category_col, value_col):
    """Horizontal bar chart, sorted smallest -> largest (our 'ranking' chart).

    Bars use a value-based colour gradient so the biggest ones stand out.
    """
    sorted_data = data.sort_values(value_col, ascending=True)
    fig = px.bar(
        sorted_data,
        x=value_col,
        y=category_col,
        orientation="h",
        color=value_col,                  # gradient colour by value
        text=value_col,
        title=f"Top {category_col} by {value_col}",
        color_continuous_scale="Tealgrn",
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)   # hide colour bar, keep it clean
    return style_chart(fig)


def make_area_chart(data, category_col, value_col):
    """Filled area chart for trends over time (first column is date/month)."""
    fig = px.area(
        data,
        x=category_col,
        y=value_col,
        markers=True,
        title=f"{value_col} Trend over {category_col}",
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    fig.update_traces(line=dict(width=3), fillcolor="rgba(46,134,222,0.25)")
    return style_chart(fig)


def make_donut_chart(data, category_col, value_col):
    """Donut (pie with a hole): how a total splits into a few parts.

    Each slice shows both its label and its percentage.
    """
    fig = px.pie(
        data,
        names=category_col,
        values=value_col,
        hole=0.55,
        title=f"{value_col} by {category_col}",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        marker=dict(line=dict(color="white", width=2)),
    )
    return style_chart(fig)


def make_treemap_chart(data, category_col, value_col):
    """Treemap (coloured rectangles): handy when there are MANY categories.

    Each item's rectangle size is its value and the colour shows how big.
    """
    fig = px.treemap(
        data,
        path=[category_col],
        values=value_col,
        color=value_col,
        title=f"{value_col} Breakdown by {category_col}",
        color_continuous_scale="Sunsetdark",
    )
    fig.update_traces(textinfo="label+value")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return style_chart(fig)


# ---------------------------------------------------------------------
# MAIN ENTRY POINTS USED BY app.py
# ---------------------------------------------------------------------

def apply_currency(fig, value_col, chart_type):
    """Add a currency symbol and short K/M value form when values are money.

    Called once at the end of build_chart. It only changes how the numbers
    are LABELLED - the bars, the hover box and the value axis get a currency
    symbol (default "$") and a compact 1.2M style value. Charts whose value
    column is not money (counts, ids ...) are returned unchanged.
    """
    if not is_currency_column(value_col):
        return fig

    symbol = detect_symbol(value_col)
    on_bar = symbol + "%{text:.2s}"        # e.g. $1.2M printed on a bar

    if chart_type == "bar_v":
        fig.update_yaxes(tickprefix=symbol)
        fig.update_traces(texttemplate=on_bar,
                          hovertemplate="%{x}: " + symbol + "%{y:.2s}<extra></extra>")
    elif chart_type == "bar_h":
        fig.update_xaxes(tickprefix=symbol)
        fig.update_traces(texttemplate=on_bar,
                          hovertemplate="%{y}: " + symbol + "%{x:.2s}<extra></extra>")
    elif chart_type == "area":
        fig.update_yaxes(tickprefix=symbol)
        fig.update_traces(hovertemplate="%{x}: " + symbol + "%{y:.2s}<extra></extra>")
    elif chart_type == "donut":
        fig.update_traces(hovertemplate="%{label}: " + symbol +
                          "%{value:.2s} (%{percent})<extra></extra>")
    elif chart_type == "treemap":
        fig.update_traces(texttemplate="%{label}<br>" + symbol + "%{value:.2s}",
                          hovertemplate="%{label}: " + symbol +
                          "%{value:.2s}<extra></extra>")
    return fig


def build_chart(intent, df):
    """Build the right chart for a query result (the function app.py calls).

    Two steps: ask pick_chart_type() which chart suits the data, then call
    the matching make_*() builder. The FIRST column is the category
    (x-axis / labels) and the SECOND column is the value (y-axis / sizes).

    Returns (figure, chart_type). figure is None when there is nothing to
    draw (e.g. a single KPI number, which app.py shows on its own);
    chart_type is kept so app.py can label the chart.
    """
    chart_type = pick_chart_type(intent, df)

    # nothing to draw, or a single KPI number handled by app.py
    if chart_type is None or chart_type == "kpi":
        return None, chart_type

    category_col = df.columns[0]
    value_col = df.columns[1]

    # send the data to the correct builder based on the chosen type
    if chart_type == "bar_v":
        figure = make_vertical_bar(df, category_col, value_col)
    elif chart_type == "bar_h":
        figure = make_horizontal_bar(df, category_col, value_col)
    elif chart_type == "area":
        figure = make_area_chart(df, category_col, value_col)
    elif chart_type == "donut":
        figure = make_donut_chart(df, category_col, value_col)
    elif chart_type == "treemap":
        figure = make_treemap_chart(df, category_col, value_col)
    else:
        # safety fallback - should not really happen
        figure = make_vertical_bar(df, category_col, value_col)

    # add currency symbol + K/M short form when the value column holds money
    figure = apply_currency(figure, value_col, chart_type)
    return figure, chart_type


def build_forecast_chart(hist_df, fc_df, dim="period", metric="value"):
    """Forecast chart: past values (solid line) plus predictions (dashed line).

    Two traces on one chart make it clear where the real data ends and the
    prediction begins. dim/metric name the x-axis and value columns.
    Returns a Plotly figure with both lines.
    """
    fig = go.Figure()

    # solid line: the real, historical numbers
    fig.add_trace(go.Scatter(
        x=hist_df[dim],
        y=hist_df[metric],
        mode="lines+markers",
        name="Historical",
        line=dict(color=CHART_COLORS[0], width=3),
        marker=dict(size=7),
        fill="tozeroy",
        fillcolor="rgba(46,134,222,0.15)",
    ))

    # dashed line: the forecast (predicted) numbers
    fig.add_trace(go.Scatter(
        x=fc_df[dim],
        y=fc_df[metric],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=CHART_COLORS[1], width=3, dash="dot"),
        marker=dict(size=9, symbol="diamond"),
    ))

    fig.update_layout(title=f"{metric} - Forecast")
    return style_chart(fig)

