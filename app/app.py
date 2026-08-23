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

st.set_page_config(
    page_title="Tark AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def read_csv_safely(path):
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
    text = open(path, "rb").read().decode("latin-1", errors="replace").replace("\x00", "")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")

def convert_excel_to_csv(excel_path):
    df = pd.read_excel(excel_path, sheet_name=0)
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path

def clean_table_name(name):
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "table"

def check_port(host, port):
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
    return "🟢" if ok else "🔴"

def format_sql(sql):
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
    st.title("🧠 Tark AI")
    st.subheader("Self-Serve Analytics Agent")
    st.write(
    )
    st.write("---")

    st.subheader("⚙️ System Status")
    services = [("HDFS NameNode", 9000), ("Hive Metastore", 9083),
                ("MySQL", 3306), ("Ollama LLM", 11434)]
    for col, (name, port) in zip(st.columns(4), services):
        ok = check_port("localhost", port)
        col.write(f"{status_icon(ok)} **{name}**")
        col.caption(f"Port {port} - {'Running' if ok else 'Stopped'}")
    st.write("---")

    # project team
    st.subheader("👥 Project Team")
    members = ["Nikhil", "Ashutosh", "Krutik", "Sreenath", "Omkar"]
    for col, name in zip(st.columns(len(members)), members):
        col.write(f"👤 **{name}**")

def render_upload():
    st.header("📤 Upload Dataset")
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
    st.success(f"✅ File saved at: `{local_path}`")

    table = clean_table_name(os.path.splitext(uploaded.name)[0])

    st.subheader("👀 Data Preview")
    preview_df = read_csv_safely(local_path)
    st.dataframe(preview_df.head(10), use_container_width=True)

    stats = [("Rows", len(preview_df)), ("Columns", len(preview_df.columns)), ("Target Table", table)]
    for col, (label, val) in zip(st.columns(3), stats):
        col.metric(label, val)
    st.write("---")

    if not st.button("🚀 Run Pipeline", use_container_width=True):
        return
    
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
            st.error(f"❌ Step failed: {label}")
            st.write(f"Return code: {r.returncode}")
            with st.expander("View Error Output", expanded=True):
                st.code("STDOUT:\n" + (r.stdout or "")[-2000:])
                st.code("STDERR:\n" + (r.stderr or "")[-2000:])
            failed = True
            break
        st.write(f"✅ Done: {label}")

    if not failed:
        progress.progress(1.0, text="✅ Pipeline complete!")
        st.success(f"🎉 Pipeline complete! Table **`{table}`** is ready in Hive.")

def render_profiling():
    st.header("📊 Data Profiling")
    st.write("Inspect data quality, column statistics and distributions for loaded tables.")
    st.write("---")

    tables = [r[0] for r in fetch_all("SELECT DISTINCT table_name FROM schema_context")]
    if not tables:
        st.info("📭 No tables found. Upload a dataset first.")
        return

    selected_table = st.selectbox("📋 Select Table to Profile", tables)
    col_info = fetch_all(
        "SELECT column_name, data_type FROM schema_context WHERE table_name=%s",
        (selected_table,))
    if not col_info:
        return

    col_df = pd.DataFrame(col_info, columns=["Column", "Data Type"])

    numeric_types = {"int", "bigint", "double", "float", "decimal", "long", "smallint"}
    num_cols = [r[0] for r in col_info if r[1].split("(")[0].lower() in numeric_types]
    str_cols = [r[0] for r in col_info if r[1].lower() == "string"]

    stats = [("📋 Total Columns", len(col_info)), ("🔢 Numeric Columns", len(num_cols)),
             ("🔡String Columns", len(str_cols)), ("📊 Data Types", col_df["Data Type"].nunique())]
    for col, (label, val) in zip(st.columns(4), stats):
        col.metric(label, val)
    st.write("---")

    st.subheader("📋 Column Schema")
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    report_path = os.path.join(DATA_DIR, f"{selected_table}_quality_report.json")
    if not os.path.exists(report_path):
        return

    import json
    st.write("---")
    st.subheader("🧹 Data Quality Report")
    with open(report_path) as f:
        report = json.load(f)

    steps = report.get("steps", [])

    def get_step(name):
        for s in steps:
            if s.get("step") == name:
                return s
        return {}
    
    duplicates_removed = get_step("Remove Duplicates").get("removed", 0)
    text_cols_cleaned = get_step("Trim Strings").get("columns_affected", 0)
    cols_renamed = get_step("Normalize Column Names").get("columns_renamed", 0)

    numeric_impute = get_step("Impute Numeric Nulls")
    numeric_cols_imputed = numeric_impute.get("columns_imputed", {})
    numeric_strategy = numeric_impute.get("strategy", "median")
    numeric_nulls_filled = sum(
        info.get("nulls_filled", 0) for info in numeric_cols_imputed.values()
    )
    cat_impute = get_step("Impute Text/Categorical Nulls")
    cat_cols_imputed = cat_impute.get("columns_imputed", {})
    cat_placeholder = cat_impute.get("placeholder", "Unknown")
    cat_nulls_filled = sum(
        info.get("nulls_filled", 0) for info in cat_cols_imputed.values()
    )
    range_step = get_step("Range Validation (rule based)")
    violations = range_step.get("violations", {})
    range_invalid = sum(info.get("invalid_values", 0) for info in violations.values())

    def _dash(v):
        return "—" if v is None else v

    rows_before = report.get("rows_before", 0)
    rows_after = report.get("rows_after", 0)
    rows_removed = rows_before - rows_after
    total_columns = report.get("columns", 0)

    total_cells = max(1, rows_before * total_columns)
    issues = duplicates_removed + numeric_nulls_filled + cat_nulls_filled + range_invalid
    score = round(max(0.0, min(100.0, 100 - issues / total_cells * 100)), 1)

    left, right = st.columns([1, 2])
    with left:
        st.metric("🏆 Data Quality Score", f"{score}%")
    with right:
        st.write("")
        st.progress(score / 100)
        if score >= 90:
            st.success("Excellent — the dataset is clean and ready to query.")
        elif score >= 75:
            st.info("Good — a few small issues were fixed during cleaning.")
        else:
            st.warning("Needs attention — several issues were found and fixed.")

    st.caption(f"{rows_before} rows in  →  {rows_after} rows kept  ({rows_removed} removed)  •  {total_columns} columns")
    st.write("")

    # simple summary table: what each cleaning step actually did
    st.markdown("#### 🧾 What the cleaning pipeline did")
    summary = pd.DataFrame({
        "Cleaning Step": [
            "1. Normalize Column Names",
            "2. Remove Duplicates",
            "3. Trim Text Columns",
            "4. Impute Numeric Nulls",
            "5. Impute Text/Categorical Nulls",
            "6. Null Out-of-Range Values",
        ],
        "Result": [
            f"{cols_renamed} columns renamed to snake_case",
            f"{duplicates_removed} duplicate rows removed",
            f"{text_cols_cleaned} text columns trimmed",
            f"{numeric_nulls_filled} nulls filled with {numeric_strategy} across {len(numeric_cols_imputed)} column(s)",
            f"{cat_nulls_filled} nulls filled with '{cat_placeholder}' across {len(cat_cols_imputed)} column(s)",
            f"{range_invalid} out-of-range values set to NULL",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    impute_rows = []
    for col_name, info in numeric_cols_imputed.items():
        impute_rows.append({
            "Column": col_name,
            "Type": "Numeric",
            "Strategy": info.get("strategy") or info.get("note") or info.get("error") or numeric_strategy,
            "Fill Value": _dash(info.get("fill_value")),
            "Nulls Filled": info.get("nulls_filled", 0),
        })
    for col_name, info in cat_cols_imputed.items():
        impute_rows.append({
            "Column": col_name,
            "Type": "Text/Categorical",
            "Strategy": info.get("strategy", "placeholder"),
            "Fill Value": info.get("fill_value", cat_placeholder),
            "Nulls Filled": info.get("nulls_filled", 0),
        })

    st.markdown("🩹 Null Imputation Detail (per column)")
    if impute_rows:
        st.caption("Exactly which columns had nulls filled, the value used, and how many rows.")
        st.dataframe(pd.DataFrame(impute_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No nulls needed imputation — every value was already present.")

    left_untouched = cat_impute.get("left_untouched", [])
    if left_untouched:
        st.caption("Left as NULL (free-text columns, not imputed): " + ", ".join(left_untouched))

    if violations:
        st.markdown("# 📈 Range Validation (out-of-range values set to NULL)")
        range_data = [
            {
                "Column": col_name,
                "Invalid Values": info.get("invalid_values", 0),
                "Lower Bound": _dash(info.get("lower_bound")),
                "Upper Bound": _dash(info.get("upper_bound")),
            }
            for col_name, info in violations.items()
        ]
        st.dataframe(pd.DataFrame(range_data), use_container_width=True, hide_index=True)

def render_metadata():
    st.header("🗂️ Metadata Explorer")
    st.write("Explore the metadata stored in MySQL that powers the query engine.")
    st.write("---")

    tab1, tab2 = st.tabs(["📋 Schema", "🔗 Relationships"])

    with tab1:
        st.subheader("📋 Tables & Columns")
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
        st.subheader("🔗 Auto-Detected Relationships")
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
    st.header("💬 Ask Analytics")
    st.write("Type your question in plain English. The ML engine detects the intent, "
             "generates Hive SQL, runs it, and returns charts and insights.")
    st.caption("ML Intent | Rule SQL | LLM Fallback | Auto Chart")

    query = st.text_input("🔍 Your question:",
                          placeholder="e.g. What is the total amount by category?")
    if not st.button("⚡ Run Query", use_container_width=True):
        return

    with st.spinner("🧠 Generating..."):
        start_time = time.time()
        r = process(query)
        total_time = round(time.time() - start_time, 2)

    st.subheader("🔬 Pipeline Trace")
    st.write(f"**Step 1 - Intent Detection:** {r['intent']}")
    validation_status = "✅ Passed" if r["valid"] else "❌ Failed"
    st.write(f"**Step 2 - Validation:** {validation_status}")
    if r["valid"]:
        engine_label = "Rule-based" if r["engine"] == "rule" else "Ollama LLM 🤖"
        st.write(f"**Step 3 - SQL Engine:** {engine_label}")
        st.write(f"**Step 4 - Execution:** Completed in {total_time}s")
    st.write("---")

    if not r["valid"]:
        st.warning(f"⚠️ {r['message']}")
        return
    if not r["sql"]:
        st.error(f"��� {r['message']}")
        return

    # generated SQL
    st.subheader("📝 Generated Hive SQL")
    st.code(format_sql(r["sql"]), language="sql")

    res = r["result"]
    if res and res["success"]:
        df = res["dataframe"]

        st.subheader("📋 Result Table")
        st.dataframe(format_dataframe(df), use_container_width=True, hide_index=True)

        # auto chart
        fig, chart = build_chart(r["intent"], df)
        if fig is not None:
            st.subheader(f"📊 Visualization - {chart.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Business Insights")
        ins_list = generate_insights(df, r["intent"])
        for ins in ins_list:
            st.write(f"- {ins}")
    elif res:
        st.error(f"❌ Query execution failed: {res['error']}")

def render_accuracy():
    st.header("🎯 Accuracy Evaluation")
    st.write("---")

    if not st.button("🧪 Run Evaluation", use_container_width=True):
        return

    from evaluation.evaluate import evaluate
    with st.spinner("🧠 Evaluating test queries..."):
        scores = evaluate()

    cards = [("🧠 Intent Accuracy", f"{scores['intent_accuracy']}%"),
             ("📝 SQL Correctness", f"{scores['sql_accuracy']}%"),
             ("🏆 Overall Accuracy", f"{scores['overall']}%")]
    for col, (label, val) in zip(st.columns(3), cards):
        col.metric(label, val)
    st.write("---")

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

    diff_scores = scores.get("difficulty_scores", {})
    if diff_scores:
        st.write("---")
        st.subheader("🎯 Per-Difficulty Breakdown")
        diff_icons = {"easy": "🟢", "medium": "🟡", "hard": "🔴", "edge": "🔵"}
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
        st.subheader("📄 Detailed Report")
        with open(report_path) as f:
            with st.expander("View Full Evaluation Report", expanded=False):
                st.code(f.read(), language="text")

def render_query_logs():
   
    st.header("📋 Query Logs")
    st.write("All queries run by users are stored in MySQL. Showing last 99.")
    st.write("---")

    rows = fetch_all(
        "SELECT id, user_query, engine, intent, execution_time_sec, "
        "row_count, status, created_at FROM query_logs ORDER BY id DESC LIMIT 100")
    if not rows:
        st.info("📭 No queries logged yet. Go to **Ask Analytics** to run your first query!")
        return

    log_df = pd.DataFrame(rows, columns=[
        "ID", "Query", "Engine", "Intent", "Time (s)", "Rows", "Status", "Date/Time"])
    
    if len(log_df) > 1:
        import plotly.express as px
        st.subheader("⚙️ Engine Usage")
        engine_counts = log_df["Engine"].value_counts().reset_index()
        engine_counts.columns = ["Engine", "Count"]
        fig = px.pie(engine_counts, names="Engine", values="Count", hole=0.5)
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write("---")

    st.subheader("📋 Full Query Log")
    st.dataframe(log_df, use_container_width=True, hide_index=True)

def render_manage_datasets():
    st.header("🗑️ Manage Datasets")
    st.write("Delete a dataset you no longer need. This removes its Hive table, the "
             "HDFS raw/cleaned copies, the local CSV + quality report, and all of its "
             "metadata (schema, business terms, KPIs and relationships).")
    st.write("---")

    from ingestion.delete_dataset import list_datasets, delete_dataset
    datasets = list_datasets()
    if not datasets:
        st.info("📭 No datasets found. Upload a dataset first.")
        return

    st.metric("📊 Datasets Loaded", len(datasets))
    selected_table = st.selectbox("🗑️ Select dataset to delete", datasets)
    st.warning(f"⚠️ This will permanently delete **{selected_table}** and everything "
               f"the pipeline created for it. This cannot be undone.")
    confirm = st.checkbox(f"Yes, permanently delete '{selected_table}'")

    if st.button("🗑️ Delete Dataset", use_container_width=True):
        if not confirm:
            st.error("❌ Please tick the confirmation box first.")
            return
        with st.spinner(f"🧹 Deleting '{selected_table}' ..."):
            summary = delete_dataset(selected_table)
        st.success(f"✅ Dataset '{selected_table}' deleted successfully.")
        removed = summary.get("local_files_removed", [])
        if removed:
            st.caption("Local files removed:")
            for path in removed:
                st.write(f"- `{path}`")
        st.balloons()

with st.sidebar:
    st.title("🧠 Tark AI")
    st.caption("Self-Serve Analytics Agent")
    st.write("---")
    PAGES = [
        "🏠  Home",
        "📤  Upload Dataset",
        "🗑️  Manage Datasets",
        "📊  Data Profiling",
        "🗂️  Metadata Explorer",
        "💬  Ask Analytics",
        "🎯  Accuracy",
        "📋  Query Logs",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.write("---")
    st.caption("CDAC DBDA Project")
    st.caption("Tech stack: Spark | Hive | ML(Random Forest) | Plotly")

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


