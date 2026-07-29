# config/config.py
# Central configuration for the Tark AI pipeline (Layers 1-3)

import os

# --- HDFS paths ---
HDFS_RAW_DIR = "hdfs://localhost:9000/tark_ai/raw"
HDFS_CLEANED_DIR = "hdfs://localhost:9000/tark_ai/cleaned"

# --- Hive ---
HIVE_DATABASE = "tark_ai"
HIVE_METASTORE_URI = "thrift://localhost:9083"

# --- MySQL metadata store ---
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "hive"
MYSQL_PASSWORD = "hivepassword"
MYSQL_DB = "tark_metadata"

# --- Local working dirs ---
# Project root = the folder that holds this config/ package. We derive it from
# this file's location so it works no matter where the project lives
# (e.g. ~/tark_ai or ~/tark_ai_2) - no hardcoded home path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Ollama (Layer 8b fallback) ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# --- Execution engine: "spark" (default, robust on ARM) or "hive" (PyHive) ---
EXECUTION_ENGINE = "spark"
HIVE_SERVER_HOST = "localhost"
HIVE_SERVER_PORT = 10000
