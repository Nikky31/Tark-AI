# config/db.py - MySQL connection helper for metadata store
# (No sys.path hack needed - install the project with `pip install -e .` so
#  `from config.config import ...` resolves as a normal package import.)

import pymysql
from config.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB


def get_connection():
    """Return a live PyMySQL connection to the tark_metadata database."""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        autocommit=True,
    )


def execute(query, params=None):
    """Run an INSERT/UPDATE/DELETE statement."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
    finally:
        conn.close()


def fetch_all(query, params=None):
    """Run a SELECT and return all rows as a list of tuples."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()
    finally:
        conn.close()
