import pandas as pd
import plotly.express as px

from analytics.money import is_currency_column, detect_symbol

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

def pick_chart_type(intent, data):
    if data is None or data.empty or len(data.columns) < 2:
        return None

    first_column = data.columns[0]
    unique_labels = data[first_column].nunique()

    if intent == "Trend":
        return "area"

    if intent == "Ranking":
        return "bar_h"

    if intent == "Comparison":
        return "bar_v"

    if intent == "Aggregation":
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
        return "kpi"
    
    return "bar_v"

def make_vertical_bar(data, category_col, value_col):
    fig = px.bar(
        data,
        x=category_col,
        y=value_col,
        color=category_col,              
        text=value_col,                   
        title=f"{value_col} by {category_col}",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(showlegend=False)  
    return style_chart(fig)


def make_horizontal_bar(data, category_col, value_col):
    sorted_data = data.sort_values(value_col, ascending=True)
    fig = px.bar(
        sorted_data,
        x=value_col,
        y=category_col,
        orientation="h",
        color=value_col,                 
        text=value_col,
        title=f"Top {category_col} by {value_col}",
        color_continuous_scale="Tealgrn",
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)  
    return style_chart(fig)


def make_area_chart(data, category_col, value_col):
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

def apply_currency(fig, value_col, chart_type):
    if not is_currency_column(value_col):
        return fig

    symbol = detect_symbol(value_col)
    on_bar = symbol + "%{text:.2s}"        

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
    chart_type = pick_chart_type(intent, df)
    if chart_type is None or chart_type == "kpi":
        return None, chart_type

    category_col = df.columns[0]
    value_col = df.columns[1]

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
        figure = make_vertical_bar(df, category_col, value_col)

    # add currency symbol
    figure = apply_currency(figure, value_col, chart_type)
    return figure, chart_type
