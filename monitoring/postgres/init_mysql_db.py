# monitoring/postgres/init_mysql_db.py
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from monitoring.configs.db_config import get_engine
import pymysql

def create_database_if_not_exists():
    """Create the fraud_monitoring database if it does not exist."""
    try:
        # Step 1: Connect to MySQL without specifying a DB
        engine_root = create_engine("mysql+pymysql://root:@localhost:3306")
        with engine_root.connect() as conn:
            conn.execute(text("CREATE DATABASE IF NOT EXISTS fraud_monitoring"))
            print("✅ [SUCCESS] Database 'fraud_monitoring' created or verified.")

        # Step 2: Connect to the actual DB
        engine = get_engine()
        with engine.connect() as conn:
            print(f"✅ [SUCCESS] Connected successfully: {engine}")
            conn.execute(text("SELECT 1"))
            print("✅ [SUCCESS] Database initialization complete.")

    except OperationalError as e:
        print(f"❌ [ERROR] Database initialization failed.\n{e}")
    except Exception as e:
        print(f"⚠️ [UNEXPECTED ERROR] {e}")

if __name__ == "__main__":
    create_database_if_not_exists()
