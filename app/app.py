# app/app.py
# -----------------------------------------------------------
# Tark AI - Self Service Analytics Agent
# CDAC DBDA Major Project
#
# Team Members : Nikhil, Ashutosh, Krutik, Sreenath, Omkar
# -----------------------------------------------------------
# Main Streamlit app. It ties together every module we built
# (ingestion, cleaning, hive, context, intent, validation, sql
# generation, execution, charts, insights, forecasting, report
# and evaluation).
#
# Layout idea (kept simple so it is easy to read):
#   - a few small helper functions at the top
#   - one render_*() function per sidebar page
#   - a sidebar menu that picks which page to show
# -----------------------------------------------------------

import os
import socket
import subprocess
import time
import re
import csv
import io

import pandas as pd
import streamlit as st

from config.config import HIVE_DATABASE, DATA_DIR
from config.db import fetch_all
from query_engine.pipeline import process
from analytics.charts import build_chart, build_forecast_chart
from analytics.insights import generate_insights

# basic page settings (must run before any other st command)
st.set_page_config(
    page_title="Tark AI - Analytics Agent",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def read_csv_safely(path):
    """Read a CSV into pandas even when the encoding, delimiter or bytes are odd.

    Three things commonly break a plain pd.read_csv on real-world files:
      1. Encoding - Excel often saves as Windows-1252 / Latin-1, not UTF-8.
      2. Delimiter - many CSVs (especially European Excel) use ';' or a tab
         instead of ',', so pandas sees one wide column and then crashes when
         a later row happens to contain a comma.
      3. UTF-16 - Excel's "Unicode Text" export is UTF-16 (a BOM plus NUL
         bytes), which otherwise fails with "line contains NUL".
    We peek at the first bytes to spot UTF-16, then try common encodings and
    let pandas auto-detect the delimiter (sep=None + the python engine sniffs
    , ; tab | ). latin-1 is last because it can decode ANY byte.
    """
    with open(path, "rb") as f:
        head = f.read(4096)
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in head:
        encodings.insert(0, "utf-16")

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError, csv.Error):
            continue

    # absolute last resort: read raw, drop NUL bytes, and parse from memory
    text = open(path, "rb").read().decode("latin-1", errors="replace").replace("\x00", "")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")


def convert_excel_to_csv(excel_path):
    """Convert an uploaded Excel file (.xlsx/.xls) into a clean UTF-8 CSV.

    The Spark ingestion pipeline only reads CSV, so we load the first sheet
    with pandas and write a plain comma-separated UTF-8 file next to it. This
    also removes all the encoding/delimiter guesswork that raw CSVs need,
    because here we control exactly how the CSV is written.
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def clean_table_name(name):
    """Make a safe table name (no spaces or odd characters) from a file name.

    HDFS paths and Hive table names cannot contain spaces or punctuation, so we
    lowercase the name and replace every run of non-alphanumeric characters
    with a single underscore. e.g. "class attendance 2026-27 batch" ->
    "class_attendance_2026_27_batch". Must match ingest.py's get_table_name.
    """
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "table"


def get_table_count():
    """Return how many distinct tables are registered in the schema context."""
    rows = fetch_all("SELECT DISTINCT table_name FROM schema_context")
    return len(rows)


def get_total_queries():
    """Return the total number of queries stored in the audit log."""
    rows = fetch_all("SELECT COUNT(*) FROM query_logs")
    return rows[0][0] if rows else 0


def get_relationship_count():
    """Return how many table relationships were auto-detected."""
    rows = fetch_all("SELECT COUNT(*) FROM relationship_context")
    return rows[0][0] if rows else 0


def check_port(host, port):
    """True if a service is listening on host:port (used for system status)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def status_icon(ok):
    """Green circle when a service is up, red circle when it is down."""
    return "\U0001f7e2" if ok else "\U0001f534"


def format_sql(sql):
    """Pretty-print the SQL by putting each major clause on its own line.

    The rule engine returns the SQL as one long line. We squeeze the extra
    spaces first, then push JOIN / FROM / WHERE / GROUP BY / ORDER BY /
    HAVING / LIMIT / ON onto new lines so it reads nicely on screen.
    """
    # first squeeze all extra spaces into single spaces
    query = " ".join(sql.split())

    # put JOIN clauses (JOIN / LEFT JOIN / INNER JOIN ...) on a new line
    query = re.sub(
        r"\s+((?:LEFT|RIGHT|INNER|FULL|CROSS|OUTER)\s+)*JOIN\s+",
        lambda m: "\n" + " ".join(m.group(0).split()) + " ",
        query, flags=re.IGNORECASE)

    # put the other main clauses on their own line too
    clauses = ["FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "ON"]
    for clause in clauses:
        pattern = r"\s+" + clause.replace(" ", r"\s+") + r"\s+"
        query = re.sub(pattern, "\n" + clause + " ", query, flags=re.IGNORECASE)

    return query


# -----------------------------------------------------------
# PAGE : HOME
# -----------------------------------------------------------

def render_home():
    """Landing page: intro, live system status, team and the 15-layer map."""
    st.title("\U0001f9e0 Tark AI")
    st.subheader("Self Service Analytics Agent")
    st.caption("Spark 3.5.1  |  Hive 3.1.3  |  ML Classifier  |  15 Layers")
    st.write(
        "Upload any CSV and ask questions in plain English. The system cleans "
        "the data, stores it in Hive, converts the question to SQL, runs it, "
        "and returns charts, insights and forecasts."
    )
    st.write("---")

    # system status (are the backend services running?)
    st.subheader("\u2699\ufe0f System Status")
    hdfs_ok = check_port("localhost", 9000)
    hive_ok = check_port("localhost", 9083)
    mysql_ok = check_port("localhost", 3306)
    ollama_ok = check_port("localhost", 11434)

    services = [
        ("HDFS NameNode", 9000, hdfs_ok),
        ("Hive Metastore", 9083, hive_ok),
        ("MySQL", 3306, mysql_ok),
        ("Ollama LLM", 11434, ollama_ok),
    ]
    scols = st.columns(4)
    for i, (name, port, ok) in enumerate(services):
        with scols[i]:
            status = "Running" if ok else "Stopped"
            st.write(f"{status_icon(ok)} **{name}**")
            st.caption(f"Port {port} - {status}")
    st.write("---")

    # project team
    st.subheader("\U0001f465 Project Team")
    members = ["Nikhil", "Ashutosh", "Krutik", "Sreenath", "Omkar"]
    tcols = st.columns(len(members))
    for i, name in enumerate(members):
        with tcols[i]:
            st.write(f"\U0001f464 **{name}**")


# -----------------------------------------------------------
# PAGE : UPLOAD DATASET
# -----------------------------------------------------------

def render_upload():
    """Upload a CSV, preview it, then run the Spark ingestion pipeline."""
    st.header("\U0001f4e4 Upload Dataset")
    st.write("Upload a CSV or Excel file. The pipeline will clean, load and build context automatically.")
    st.write("---")

    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return

    # save the uploaded file locally so Spark jobs can read it
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, uploaded.name)
    with open(local_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Spark's pipeline only reads CSV, so convert Excel uploads to a UTF-8 CSV
    if local_path.lower().endswith((".xlsx", ".xls")):
        local_path = convert_excel_to_csv(local_path)
    st.success(f"\u2705 File saved at: `{local_path}`")

    table = clean_table_name(os.path.splitext(uploaded.name)[0])

    st.subheader("\U0001f440 Data Preview")
    preview_df = read_csv_safely(local_path)
    st.dataframe(preview_df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(preview_df))
    with col2:
        st.metric("Columns", len(preview_df.columns))
    with col3:
        st.metric("Target Table", table)
    st.write("---")

    if not st.button("\U0001f680 Run Pipeline", use_container_width=True):
        return

    # each pipeline step is a separate spark-submit job
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    steps = [
        ("Layer 1+2: Ingest + Clean", ["spark-submit", base + "/ingestion/ingest.py", local_path]),
        ("Layer 2: Deep Cleaning", ["spark-submit", base + "/ingestion/clean.py", table]),
        ("Layer 3: Create Hive Table", ["spark-submit", base + "/ingestion/hive_loader.py", table]),
        ("Layer 5: Build Context", ["spark-submit", base + "/context/context_manager.py"]),
        ("Layer 4: Discover Relationships", ["spark-submit", base + "/context/relationships.py"]),
    ]

    progress = st.progress(0, text="Starting pipeline...")
    failed = False
    for idx, (label, cmd) in enumerate(steps):
        env = os.environ.copy()
        env["PYSPARK_DRIVER_PYTHON"] = "python3"
        env.pop("PYSPARK_DRIVER_PYTHON_OPTS", None)

        progress.progress((idx) / len(steps), text=f"Running: {label}")
        with st.spinner(f"Running -> {label}"):
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if r.returncode != 0:
            st.error(f"\u274c Step failed: {label}")
            st.write(f"Return code: {r.returncode}")
            with st.expander("View Error Output", expanded=True):
                st.code("STDOUT:\n" + (r.stdout or "")[-2000:])
                st.code("STDERR:\n" + (r.stderr or "")[-2000:])
            failed = True
            break
        else:
            st.write(f"\u2705 Done: {label}")

    if not failed:
        progress.progress(1.0, text="\u2705 Pipeline complete!")
        st.success(f"\U0001f389 Pipeline complete! Table **`{table}`** is ready in Hive.")
        st.balloons()


# -----------------------------------------------------------
# PAGE : DATA PROFILING
# -----------------------------------------------------------

def render_profiling():
    """Show column stats, data-type split and the cleaning quality report."""
    st.header("\U0001f4ca Data Profiling")
    st.write("Inspect data quality, column statistics and distributions for loaded tables.")
    st.write("---")

    schema_rows_prof = fetch_all("SELECT DISTINCT table_name FROM schema_context")
    tables_prof = [r[0] for r in schema_rows_prof]

    if not tables_prof:
        st.info("\U0001f4ed No tables found. Upload a dataset first.")
        return

    selected_table = st.selectbox("\U0001f4cb Select Table to Profile", tables_prof)
    col_info = fetch_all(
        "SELECT column_name, data_type FROM schema_context WHERE table_name=%s",
        (selected_table,))
    if not col_info:
        return

    col_df = pd.DataFrame(col_info, columns=["Column", "Data Type"])

    # summary numbers
    numeric_types = {"int", "bigint", "double", "float", "decimal", "long", "smallint"}
    num_cols = [r[0] for r in col_info if r[1].split("(")[0].lower() in numeric_types]
    str_cols = [r[0] for r in col_info if r[1].lower() == "string"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("\U0001f4cb Total Columns", len(col_info))
    with c2:
        st.metric("\U0001f522 Numeric Columns", len(num_cols))
    with c3:
        st.metric("\U0001f4dd String Columns", len(str_cols))
    with c4:
        st.metric("\U0001f4ca Data Types", col_df["Data Type"].nunique())
    st.write("---")

    # data type distribution (donut chart)
    st.subheader("\U0001f4ca Data Type Distribution")
    import plotly.express as px
    type_counts = col_df["Data Type"].value_counts().reset_index()
    type_counts.columns = ["Data Type", "Count"]
    fig_types = px.pie(type_counts, names="Data Type", values="Count", hole=0.5)
    fig_types.update_layout(height=300)
    st.plotly_chart(fig_types, use_container_width=True)
    st.write("---")

    # column schema table
    st.subheader("\U0001f4cb Column Schema")
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    # data quality report if it exists
    report_path = os.path.join(DATA_DIR, f"{selected_table}_quality_report.json")
    if os.path.exists(report_path):
        import json
        st.write("---")
        st.subheader("\U0001f9f9 Data Quality Report")
        with open(report_path) as f:
            report = json.load(f)

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.metric("Rows Before Cleaning", report.get("rows_before", "N/A"))
        with qc2:
            st.metric("Rows After Cleaning", report.get("rows_after", "N/A"))
        with qc3:
            removed = report.get("rows_before", 0) - report.get("rows_after", 0)
            st.metric("Rows Removed", removed)

        # cleaning steps
        steps = report.get("steps", [])
        for step_info in steps:
            st.write(f"\u2705 {step_info.get('step', '')}")

        # outlier info
        outlier_step = next((s for s in steps if s.get("step") == "Outlier Detection (IQR)"), None)
        if outlier_step and outlier_step.get("outliers"):
            st.subheader("\U0001f4c8 Outlier Capping Summary")
            outlier_data = []
            for col_name, info in outlier_step["outliers"].items():
                outlier_data.append({
                    "Column": col_name,
                    "Outliers Capped": info["outliers_capped"],
                    "Lower Bound": info["lower_bound"],
                    "Upper Bound": info["upper_bound"],
                })
            st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)


# -----------------------------------------------------------
# PAGE : METADATA EXPLORER
# -----------------------------------------------------------

def render_metadata():
    """Browse the MySQL metadata (schema, relationships, business terms, KPIs)."""
    st.header("\U0001f5c2\ufe0f Metadata Explorer")
    st.write("Explore the metadata stored in MySQL that powers the query engine.")
    st.write("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001f4cb Schema", "\U0001f517 Relationships", "\U0001f4d6 Business Terms", "\U0001f3af KPIs"
    ])

    with tab1:
        st.subheader("\U0001f4cb Tables & Columns")
        schema_data = fetch_all("SELECT table_name, column_name, data_type FROM schema_context")
        if schema_data:
            schema_df = pd.DataFrame(schema_data, columns=["Table", "Column", "Type"])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tables", schema_df["Table"].nunique())
            with c2:
                st.metric("Total Columns", len(schema_df))
            with c3:
                st.metric("Data Types", schema_df["Type"].nunique())
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        else:
            st.info("No schema data found. Upload a dataset first.")

    with tab2:
        st.subheader("\U0001f517 Auto-Detected Relationships")
        rel = fetch_all("SELECT parent_table, child_table, join_key, confidence FROM relationship_context")
        if rel:
            rel_df = pd.DataFrame(rel, columns=["Parent Table", "Child Table", "Join Key", "Confidence %"])
            st.metric("Relationships Found", len(rel_df))
            # show each relationship as a readable line
            for _, row in rel_df.iterrows():
                st.write(f"- **{row['Parent Table']}**.{row['Join Key']} <-> **{row['Child Table']}**.{row['Join Key']} (confidence: {row['Confidence %']}%)")
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships detected yet.")

    with tab3:
        st.subheader("\U0001f4d6 Business Term Mappings")
        biz = fetch_all("SELECT business_term, table_name, column_name, aggregation FROM business_context")
        if biz:
            st.dataframe(pd.DataFrame(biz, columns=["Business Term", "Table", "Column", "Aggregation"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No business terms defined yet.")

    with tab4:
        st.subheader("\U0001f3af KPI Definitions")
        kpis = fetch_all("SELECT kpi_name, table_name, expression FROM kpi_context")
        if kpis:
            st.dataframe(pd.DataFrame(kpis, columns=["KPI Name", "Table", "Expression"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No KPIs defined yet.")


# -----------------------------------------------------------
# PAGE : ASK ANALYTICS
# -----------------------------------------------------------

def render_ask():
    """Natural-language question -> intent -> SQL -> result, chart, insights, PDF."""
    st.header("\U0001f4ac Ask Analytics")
    st.write("Type your question in plain English. The ML engine detects the intent, "
             "generates Hive SQL, runs it, and returns charts and insights.")
    st.caption("ML Intent | Rule SQL | LLM Fallback | Auto Chart")

    query = st.text_input("\U0001f50d Your question:",
                          placeholder="e.g. What is the total amount by category?")

    if not st.button("\u26a1 Run Query", use_container_width=True):
        return

    with st.spinner("\U0001f9e0 Processing your question through the pipeline..."):
        start_time = time.time()
        r = process(query)
        total_time = round(time.time() - start_time, 2)

    # pipeline trace - shows how the layers work together
    st.subheader("\U0001f52c Pipeline Trace")
    st.write(f"**Step 1 - Intent Detection:** {r['intent']}")
    validation_status = "\u2705 Passed" if r["valid"] else "\u274c Failed"
    st.write(f"**Step 2 - Validation:** {validation_status}")
    if r["valid"]:
        engine_label = "Rule-based \u26a1" if r["engine"] == "rule" else "Ollama LLM \U0001f916"
        st.write(f"**Step 3 - SQL Engine:** {engine_label}")
        st.write(f"**Step 4 - Execution:** Completed in {total_time}s")

    st.write("---")

    if not r["valid"]:
        st.warning(f"\u26a0\ufe0f {r['message']}")
        return
    if not r["sql"]:
        st.error(f"\u274c {r['message']}")
        return

    # generated SQL
    st.subheader("\U0001f4dd Generated Hive SQL")
    st.code(format_sql(r["sql"]), language="sql")

    res = r["result"]
    if res and res["success"]:
        df = res["dataframe"]

        # result table
        st.subheader("\U0001f4cb Result Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # chart
        fig, chart = build_chart(r["intent"], df)
        if fig is not None:
            chart_name = chart.replace("_", " ").title()
            st.subheader(f"\U0001f4ca Visualization - {chart_name}")
            st.plotly_chart(fig, use_container_width=True)

        # insights
        st.subheader("\U0001f4a1 Business Insights")
        ins_list = generate_insights(df, r["intent"])
        for ins in ins_list:
            st.write(f"- {ins}")

        st.write("---")

        # PDF report
        # We build the PDF right here and show a single download
        # button. We do NOT use a separate "Generate" button on
        # purpose: Streamlit reruns the whole page on every click,
        # so a second button would make this results block
        # disappear and the download would never show up.
        st.subheader("\U0001f4c4 Download Report")
        from analytics.report import generate_pdf
        pdf_path = generate_pdf(query, r["sql"], df, ins_list, fig=fig)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                "\u2b07\ufe0f Download PDF Report",
                data=pdf_file,
                file_name="tark_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    elif res:
        st.error(f"\u274c Query execution failed: {res['error']}")


# -----------------------------------------------------------
# PAGE : INSIGHTS & FORECASTING
# -----------------------------------------------------------

def render_forecasting():
    """Pick a table/metric and forecast future values with Spark MLlib."""
    st.header("\U0001f4c8 Insights & Forecasting")
    st.write("Select a table and metric to forecast future values using Spark MLlib.")
    st.write("---")

    schema_rows = fetch_all("SELECT DISTINCT table_name FROM schema_context")
    tables = [r[0] for r in schema_rows]

    if not tables:
        st.info("\U0001f4ed No tables found. Upload a dataset first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        table = st.selectbox("\U0001f4cb Select Table", tables)
    with col2:
        all_cols = [r[0] for r in fetch_all(
            "SELECT column_name FROM schema_context WHERE table_name=%s", (table,))]

    col3, col4 = st.columns(2)
    with col3:
        dim = st.selectbox("\U0001f4c5 Order / Dimension Column", all_cols)
    with col4:
        metric = st.selectbox("\U0001f4ca Metric to Forecast", all_cols)

    periods = st.slider("\U0001f52e Future periods to predict", 1, 12, 3)
    st.write("---")

    if not st.button("\U0001f680 Run Forecast", use_container_width=True):
        return

    from analytics.forecast import forecast_metric
    with st.spinner("\U0001f9e0 Training model using Spark MLlib..."):
        hist, fc, metrics = forecast_metric(table, dim, metric, periods)

    if not fc:
        st.warning("\u26a0\ufe0f Not enough data to produce a meaningful forecast.")
        return

    h = pd.DataFrame(hist)
    f = pd.DataFrame(fc)

    # forecast chart
    fig = build_forecast_chart(h, f)
    st.subheader("\U0001f4c8 Forecast Chart")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Historical Points", len(h))
    with c2:
        st.metric("Forecast Points", len(f))
    with c3:
        if len(f) > 0:
            trend = "\U0001f4c8 Up" if f["value"].iloc[-1] > h["value"].iloc[-1] else "\U0001f4c9 Down"
            st.metric("Trend Direction", trend)

    # model evaluation numbers
    if metrics:
        st.subheader("\U0001f52c Model Evaluation")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("\U0001f4d0 RMSE", metrics.get("rmse", "N/A"))
        with mc2:
            st.metric("\U0001f4cf MAE", metrics.get("mae", "N/A"))
        with mc3:
            r2_val = metrics.get("r2", 0)
            st.metric("\U0001f4ca R2 Score", f"{r2_val:.4f}")
        with mc4:
            st.metric("\U0001f4c8 Training Points", metrics.get("training_points", "N/A"))

        with st.expander("\U0001f4cb Model Details"):
            st.write(f"**Intercept:** {metrics.get('intercept', 'N/A')}")
            st.write(f"**Coefficients:** {metrics.get('coefficients', 'N/A')}")
            st.write(f"**Model:** Linear Regression (Spark MLlib)")
            r2 = metrics.get('r2', 0)
            if r2 > 0.8:
                st.success("\u2705 Strong model fit (R2 > 0.8)")
            elif r2 > 0.5:
                st.warning("\u26a0\ufe0f Moderate model fit (0.5 < R2 < 0.8)")
            else:
                st.error("\u274c Weak model fit (R2 < 0.5). Consider non-linear models.")

        st.subheader("\U0001f52e Forecast Values")
        st.dataframe(f, use_container_width=True, hide_index=True)


# -----------------------------------------------------------
# PAGE : ACCURACY
# -----------------------------------------------------------

def render_accuracy():
    """Run the Layer 10 evaluation and show accuracy scorecards and gauges."""
    st.header("\U0001f3af Accuracy Evaluation")
    st.write("Run the evaluation framework (Layer 10) to measure intent detection "
             "accuracy and SQL correctness on 50 test queries across 4 difficulty levels.")
    st.caption("ML Classifier | 50 Test Cases | Confusion Matrix")
    st.write("---")

    if not st.button("\U0001f9ea Run Evaluation", use_container_width=True):
        return

    from evaluation.evaluate import evaluate
    with st.spinner("\U0001f9e0 Evaluating test queries..."):
        scores = evaluate()

    # score cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("\U0001f9e0 Intent Accuracy", f"{scores['intent_accuracy']}%")
    with c2:
        st.metric("\U0001f4dd SQL Correctness", f"{scores['sql_accuracy']}%")
    with c3:
        st.metric("\U0001f3c6 Overall Accuracy", f"{scores['overall']}%")
    st.write("---")

    # gauge charts for accuracy
    import plotly.graph_objects as go
    gauge_cols = st.columns(3)
    labels = ["Intent Accuracy", "SQL Correctness", "Overall"]
    keys = ["intent_accuracy", "sql_accuracy", "overall"]

    for col, label, key in zip(gauge_cols, labels, keys):
        with col:
            val = scores[key]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label},
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}},
            ))
            fig.update_layout(height=260)
            st.plotly_chart(fig, use_container_width=True)

    # difficulty breakdown
    diff_scores = scores.get("difficulty_scores", {})
    if diff_scores:
        st.write("---")
        st.subheader("\U0001f3af Per-Difficulty Breakdown")
        diff_icons = {"easy": "\U0001f7e2", "medium": "\U0001f7e1", "hard": "\U0001f534", "edge": "\U0001f535"}
        dcols = st.columns(4)
        for i, diff in enumerate(["easy", "medium", "hard", "edge"]):
            d = diff_scores.get(diff, {})
            if d.get("total", 0) > 0:
                i_pct = round(d["intent_ok"] / d["total"] * 100, 1)
                s_pct = round(d["sql_ok"] / d["total"] * 100, 1)
                icon = diff_icons[diff]
                with dcols[i]:
                    st.write(f"{icon} **{diff.upper()}**")
                    st.caption(f"{d['total']} cases")
                    st.write(f"Intent: **{i_pct}%**")
                    st.write(f"SQL: **{s_pct}%**")

    # detailed report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation", "accuracy_report.txt")
    if os.path.exists(report_path):
        st.write("---")
        st.subheader("\U0001f4c4 Detailed Report")
        with open(report_path) as f:
            with st.expander("View Full Evaluation Report", expanded=False):
                st.code(f.read(), language="text")


# -----------------------------------------------------------
# PAGE : QUERY LOGS
# -----------------------------------------------------------

def render_query_logs():
    """Audit trail: last 100 logged queries with summary metrics and charts."""
    st.header("\U0001f4cb Query Logs (Audit Trail)")
    st.write("All queries run by users are stored in MySQL. Showing last 100.")
    st.write("---")

    rows = fetch_all(
        "SELECT id, user_query, engine, intent, execution_time_sec, "
        "row_count, status, created_at FROM query_logs ORDER BY id DESC LIMIT 100")

    if not rows:
        st.info("\U0001f4ed No queries logged yet. Go to **Ask Analytics** to run your first query!")
        return

    log_df = pd.DataFrame(rows, columns=[
        "ID", "Query", "Engine", "Intent", "Time (s)", "Rows", "Status", "Date/Time"])

    # summary numbers
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("\U0001f4ca Total Queries", len(log_df))
    with c2:
        rule_count = len(log_df[log_df["Engine"] == "rule"]) if "Engine" in log_df.columns else 0
        st.metric("\u26a1 Rule-Based", rule_count)
    with c3:
        llm_count = len(log_df[log_df["Engine"] != "rule"]) if "Engine" in log_df.columns else 0
        st.metric("\U0001f916 LLM Fallback", llm_count)
    with c4:
        success_count = len(log_df[log_df["Status"] == "success"]) if "Status" in log_df.columns else 0
        st.metric("\u2705 Successful", success_count)
    st.write("---")

    # charts - engine usage and intent distribution
    if len(log_df) > 1:
        import plotly.express as px
        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.subheader("\u2699\ufe0f Engine Usage")
            engine_counts = log_df["Engine"].value_counts().reset_index()
            engine_counts.columns = ["Engine", "Count"]
            fig = px.pie(engine_counts, names="Engine", values="Count", hole=0.5)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with chart_cols[1]:
            st.subheader("\U0001f9e0 Intent Distribution")
            intent_counts = log_df["Intent"].value_counts().reset_index()
            intent_counts.columns = ["Intent", "Count"]
            fig2 = px.bar(intent_counts, x="Count", y="Intent", orientation="h", text="Count")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.write("---")

    # full log table
    st.subheader("\U0001f4cb Full Query Log")
    st.dataframe(log_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------
# PAGE : MANAGE DATASETS (delete)
# -----------------------------------------------------------

def render_manage_datasets():
    """Delete a dataset when the workspace gets cluttered with too many CSVs.

    Removes the Hive table, the HDFS raw/cleaned copies, the local CSV +
    quality report, and all of the dataset's MySQL metadata. A confirmation
    checkbox guards against accidental deletes.
    """
    st.header("\U0001f5d1\ufe0f Manage Datasets")
    st.write("Delete a dataset you no longer need. This removes its Hive table, the "
             "HDFS raw/cleaned copies, the local CSV + quality report, and all of its "
             "metadata (schema, business terms, KPIs and relationships).")
    st.write("---")

    from ingestion.delete_dataset import list_datasets, delete_dataset
    datasets = list_datasets()

    if not datasets:
        st.info("\U0001f4ed No datasets found. Upload a dataset first.")
        return

    st.metric("\U0001f4ca Datasets Loaded", len(datasets))
    selected_table = st.selectbox("\U0001f5d1\ufe0f Select dataset to delete", datasets)

    st.warning(f"\u26a0\ufe0f This will permanently delete **{selected_table}** and everything "
               f"the pipeline created for it. This cannot be undone.")

    confirm = st.checkbox(f"Yes, permanently delete '{selected_table}'")

    if st.button("\U0001f5d1\ufe0f Delete Dataset", use_container_width=True):
        if not confirm:
            st.error("\u274c Please tick the confirmation box first.")
            return
        with st.spinner(f"\U0001f9f9 Deleting '{selected_table}' ..."):
            summary = delete_dataset(selected_table)
        st.success(f"\u2705 Dataset '{selected_table}' deleted successfully.")
        removed = summary.get("local_files_removed", [])
        if removed:
            st.caption("Local files removed:")
            for path in removed:
                st.write(f"- `{path}`")
        st.balloons()


# -----------------------------------------------------------
# SIDEBAR (navigation menu) + PAGE DISPATCH
# -----------------------------------------------------------

with st.sidebar:
    st.title("\U0001f9e0 Tark AI")
    st.caption("Self Service Analytics Agent  (v2.0)")
    st.write("---")

    PAGES = [
        "\U0001f3e0  Home",
        "\U0001f4e4  Upload Dataset",
        "\U0001f5d1\ufe0f  Manage Datasets",
        "\U0001f4ca  Data Profiling",
        "\U0001f5c2\ufe0f  Metadata Explorer",
        "\U0001f4ac  Ask Analytics",
        "\U0001f4c8  Insights & Forecasting",
        "\U0001f3af  Accuracy",
        "\U0001f4cb  Query Logs",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.write("---")
    st.caption("CDAC DBDA Major Project")
    st.caption("Tech stack: Spark | Hive | ML | Plotly")

# map each sidebar page to the function that draws it
PAGE_RENDERERS = {
    PAGES[0]: render_home,
    PAGES[1]: render_upload,
    PAGES[2]: render_manage_datasets,
    PAGES[3]: render_profiling,
    PAGES[4]: render_metadata,
    PAGES[5]: render_ask,
    PAGES[6]: render_forecasting,
    PAGES[7]: render_accuracy,
    PAGES[8]: render_query_logs,
}
PAGE_RENDERERS[page]()

