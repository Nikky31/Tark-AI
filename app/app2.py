# app/app.py
# Tark AI - Self Service Analytics Agent  (CDAC DBDA Major Project)
# Team: Nikhil, Ashutosh, Krutik, Sreenath, Omkar
# One render_*() function per sidebar page, chosen by the sidebar menu.

import os
import socket
import subprocess
import time
import re
import csv
import io

import pandas as pd
import streamlit as st

from config.config import DATA_DIR
from config.db import fetch_all
from query_engine.pipeline import process
from analytics.charts import build_chart
from analytics.insights import generate_insights
from analytics.money import format_dataframe

# basic page settings (must run before any other st command)
st.set_page_config(
    page_title="Tark AI - Analytics Agent",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="expanded",
)

def read_csv_safely(path):
    """Read a CSV even when the encoding/delimiter is odd (or the file is UTF-16)."""
    with open(path, "rb") as f:
        head = f.read(4096)
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in head:
        encodings.insert(0, "utf-16")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError, csv.Error):
            continue
    # last resort: drop NUL bytes and parse from memory
    text = open(path, "rb").read().decode("latin-1", errors="replace").replace("\x00", "")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")

def convert_excel_to_csv(excel_path):
    """Convert an uploaded Excel file into a clean UTF-8 CSV (Spark reads CSV only)."""
    df = pd.read_excel(excel_path, sheet_name=0)
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path

def clean_table_name(name):
    """Make a safe table name (no spaces/punctuation). Must match ingest.py's version."""
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "table"

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
    """Pretty-print SQL by putting each major clause on its own line."""
    query = " ".join(sql.split())
    query = re.sub(
        r"\s+((?:LEFT|RIGHT|INNER|FULL|CROSS|OUTER)\s+)*JOIN\s+",
        lambda m: "\n" + " ".join(m.group(0).split()) + " ",
        query, flags=re.IGNORECASE)
    for clause in ["FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "ON"]:
        pattern = r"\s+" + clause.replace(" ", r"\s+") + r"\s+"
        query = re.sub(pattern, "\n" + clause + " ", query, flags=re.IGNORECASE)
    return query

def render_home():
    """Landing page: intro text, live system status and the project team."""
    st.title("\U0001f9e0 Tark AI")
    st.subheader("Self Service Analytics Agent")
    st.caption("Spark 3.5.1  |  Hive 3.1.3  |  ML Classifier  |  15 Layers")
    st.write(
        "Upload any CSV and ask questions in plain English. The system cleans "
        "the data, stores it in Hive, converts the question to SQL, runs it, "
        "and returns tables, charts and insights."
    )
    st.write("---")

    # system status - are the backend services running?
    st.subheader("\u2699\ufe0f System Status")
    services = [("HDFS NameNode", 9000), ("Hive Metastore", 9083),
                ("MySQL", 3306), ("Ollama LLM", 11434)]
    for col, (name, port) in zip(st.columns(4), services):
        ok = check_port("localhost", port)
        col.write(f"{status_icon(ok)} **{name}**")
        col.caption(f"Port {port} - {'Running' if ok else 'Stopped'}")
    st.write("---")

    # project team
    st.subheader("\U0001f465 Project Team")
    members = ["Nikhil", "Ashutosh", "Krutik", "Sreenath", "Omkar"]
    for col, name in zip(st.columns(len(members)), members):
        col.write(f"\U0001f464 **{name}**")

def render_upload():
    """Upload a CSV/Excel file, preview it, then run the Spark ingestion pipeline."""
    st.header("\U0001f4e4 Upload Dataset")
    st.write("Upload a CSV or Excel file. The pipeline will clean, load and build context automatically.")
    st.write("---")

    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if not uploaded:
        return

    # save the uploaded file locally so Spark jobs can read it
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, uploaded.name)
    with open(local_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Spark's pipeline only reads CSV, so convert Excel uploads to a UTF-8 CSV
    if local_path.lower().endswith(".xlsx"):
        local_path = convert_excel_to_csv(local_path)
    st.success(f"\u2705 File saved at: `{local_path}`")

    table = clean_table_name(os.path.splitext(uploaded.name)[0])

    st.subheader("\U0001f440 Data Preview")
    preview_df = read_csv_safely(local_path)
    st.dataframe(preview_df.head(10), use_container_width=True)

    stats = [("Rows", len(preview_df)), ("Columns", len(preview_df.columns)), ("Target Table", table)]
    for col, (label, val) in zip(st.columns(3), stats):
        col.metric(label, val)
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

        progress.progress(idx / len(steps), text=f"Running: {label}")
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
        st.write(f"\u2705 Done: {label}")

    if not failed:
        progress.progress(1.0, text="\u2705 Pipeline complete!")
        st.success(f"\U0001f389 Pipeline complete! Table **`{table}`** is ready in Hive.")
        st.balloons()

def render_profiling():
    """Show column stats, data-type split and the cleaning quality report."""
    st.header("\U0001f4ca Data Profiling")
    st.write("Inspect data quality, column statistics and distributions for loaded tables.")
    st.write("---")

    tables = [r[0] for r in fetch_all("SELECT DISTINCT table_name FROM schema_context")]
    if not tables:
        st.info("\U0001f4ed No tables found. Upload a dataset first.")
        return

    selected_table = st.selectbox("\U0001f4cb Select Table to Profile", tables)
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

    stats = [("\U0001f4cb Total Columns", len(col_info)), ("\U0001f522 Numeric Columns", len(num_cols)),
             ("\U0001f4dd String Columns", len(str_cols)), ("\U0001f4ca Data Types", col_df["Data Type"].nunique())]
    for col, (label, val) in zip(st.columns(4), stats):
        col.metric(label, val)
    st.write("---")

    # column schema table
    st.subheader("\U0001f4cb Column Schema")
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    # data quality report (only if the cleaning step wrote one)
    report_path = os.path.join(DATA_DIR, f"{selected_table}_quality_report.json")
    if not os.path.exists(report_path):
        return

    import json
    st.write("---")
    st.subheader("\U0001f9f9 Data Quality Report")
    with open(report_path) as f:
        report = json.load(f)

    removed = report.get("rows_before", 0) - report.get("rows_after", 0)
    quality = [("Rows Before Cleaning", report.get("rows_before", "N/A")),
               ("Rows After Cleaning", report.get("rows_after", "N/A")), ("Rows Removed", removed)]
    for col, (label, val) in zip(st.columns(3), quality):
        col.metric(label, val)

    steps = report.get("steps", [])
    for step_info in steps:
        st.write(f"\u2705 {step_info.get('step', '')}")

    outlier_step = next((s for s in steps if s.get("step") == "Outlier Detection (IQR)"), None)
    if outlier_step and outlier_step.get("outliers"):
        st.subheader("\U0001f4c8 Outlier Capping Summary")
        outlier_data = [
            {"Column": c, "Outliers Capped": info["outliers_capped"],
             "Lower Bound": info["lower_bound"], "Upper Bound": info["upper_bound"]}
            for c, info in outlier_step["outliers"].items()
        ]
        st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)

def render_metadata():
    """Browse the MySQL metadata: table schema and auto-detected relationships."""
    st.header("\U0001f5c2\ufe0f Metadata Explorer")
    st.write("Explore the metadata stored in MySQL that powers the query engine.")
    st.write("---")

    tab1, tab2 = st.tabs(["\U0001f4cb Schema", "\U0001f517 Relationships"])

    with tab1:
        st.subheader("\U0001f4cb Tables & Columns")
        schema_data = fetch_all("SELECT table_name, column_name, data_type FROM schema_context")
        if schema_data:
            schema_df = pd.DataFrame(schema_data, columns=["Table", "Column", "Type"])
            stats = [("Tables", schema_df["Table"].nunique()), ("Total Columns", len(schema_df)),
                     ("Data Types", schema_df["Type"].nunique())]
            for col, (label, val) in zip(st.columns(3), stats):
                col.metric(label, val)
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        else:
            st.info("No schema data found. Upload a dataset first.")

    with tab2:
        st.subheader("\U0001f517 Auto-Detected Relationships")
        rel = fetch_all("SELECT parent_table, child_table, join_key, confidence FROM relationship_context")
        if rel:
            rel_df = pd.DataFrame(rel, columns=["Parent Table", "Child Table", "Join Key", "Confidence %"])
            st.metric("Relationships Found", len(rel_df))
            for _, row in rel_df.iterrows():
                st.write(f"- **{row['Parent Table']}**.{row['Join Key']} <-> **{row['Child Table']}**.{row['Join Key']} (confidence: {row['Confidence %']}%)")
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships detected yet.")

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

        st.subheader("\U0001f4cb Result Table")
        st.dataframe(format_dataframe(df), use_container_width=True, hide_index=True)

        # auto chart
        fig, chart = build_chart(r["intent"], df)
        if fig is not None:
            st.subheader(f"\U0001f4ca Visualization - {chart.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

        # business insights
        st.subheader("\U0001f4a1 Business Insights")
        ins_list = generate_insights(df, r["intent"])
        for ins in ins_list:
            st.write(f"- {ins}")
    elif res:
        st.error(f"\u274c Query execution failed: {res['error']}")

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
    cards = [("\U0001f9e0 Intent Accuracy", f"{scores['intent_accuracy']}%"),
             ("\U0001f4dd SQL Correctness", f"{scores['sql_accuracy']}%"),
             ("\U0001f3c6 Overall Accuracy", f"{scores['overall']}%")]
    for col, (label, val) in zip(st.columns(3), cards):
        col.metric(label, val)
    st.write("---")

    # gauge charts
    import plotly.graph_objects as go
    labels = ["Intent Accuracy", "SQL Correctness", "Overall"]
    keys = ["intent_accuracy", "sql_accuracy", "overall"]
    for col, label, key in zip(st.columns(3), labels, keys):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scores[key],
            title={"text": label},
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}},
        ))
        fig.update_layout(height=260)
        col.plotly_chart(fig, use_container_width=True)

    # per-difficulty breakdown
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
                with dcols[i]:
                    st.write(f"{diff_icons[diff]} **{diff.upper()}**")
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
    stats = [("\U0001f4ca Total Queries", len(log_df)), ("\u26a1 Rule-Based", len(log_df[log_df["Engine"] == "rule"])),
             ("\U0001f916 LLM Fallback", len(log_df[log_df["Engine"] != "rule"])), ("\u2705 Successful", len(log_df[log_df["Status"] == "success"]))]
    for col, (label, val) in zip(st.columns(4), stats):
        col.metric(label, val)
    st.write("---")

    # charts - engine usage and intent distribution
    if len(log_df) > 1:
        import plotly.express as px
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("\u2699\ufe0f Engine Usage")
            engine_counts = log_df["Engine"].value_counts().reset_index()
            engine_counts.columns = ["Engine", "Count"]
            fig = px.pie(engine_counts, names="Engine", values="Count", hole=0.5)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("\U0001f9e0 Intent Distribution")
            intent_counts = log_df["Intent"].value_counts().reset_index()
            intent_counts.columns = ["Intent", "Count"]
            fig2 = px.bar(intent_counts, x="Count", y="Intent", orientation="h", text="Count")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        st.write("---")

    st.subheader("\U0001f4cb Full Query Log")
    st.dataframe(log_df, use_container_width=True, hide_index=True)

def render_manage_datasets():
    """Delete a dataset (Hive table, HDFS copies, local files and MySQL metadata)."""
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
    PAGES[6]: render_accuracy,
    PAGES[7]: render_query_logs,
}
PAGE_RENDERERS[page]()

