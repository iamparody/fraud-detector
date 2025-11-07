# monitoring/configs/db_config.py
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def get_engine():
    try:
        # MySQL connection string for XAMPP
        username = "root"        # default XAMPP user
        password = ""            # leave empty unless you set one
        host = "localhost"
        port = 3306
        database = "fraud_monitoring"

        connection_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_url)
        print(f"✅ [SUCCESS] Connected successfully: {engine}")
        return engine
    except SQLAlchemyError as e:
        print(f"❌ [ERROR] Database connection failed: {e}")
        raise
