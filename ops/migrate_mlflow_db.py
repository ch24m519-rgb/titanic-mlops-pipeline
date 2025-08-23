import psycopg2
import os

# Load DB connection settings (these should match your docker-compose.yml)
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "mlflow")
DB_USER = os.getenv("POSTGRES_USER", "mlflow")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "mlflow")

MIGRATION_SQL = """
-- Make experiment_id BIGINT instead of INTEGER
ALTER TABLE experiments ALTER COLUMN experiment_id TYPE BIGINT USING experiment_id::bigint;
ALTER TABLE runs ALTER COLUMN experiment_id TYPE BIGINT USING experiment_id::bigint;

-- Some MLflow tables reference run_uuid; keep them safe
ALTER TABLE metrics ALTER COLUMN run_uuid TYPE VARCHAR(36);
ALTER TABLE params ALTER COLUMN run_uuid TYPE VARCHAR(36);
ALTER TABLE tags ALTER COLUMN run_uuid TYPE VARCHAR(36);
"""

def migrate():
    print(f"[INFO] Connecting to Postgres at {DB_HOST}:{DB_PORT}, DB={DB_NAME}")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    conn.autocommit = True
    cur = conn.cursor()
    try:
        print("[INFO] Running migration...")
        cur.execute(MIGRATION_SQL)
        print("Migration complete: experiment_id is now BIGINT")
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    migrate()

