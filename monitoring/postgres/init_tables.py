# monitoring/postgres/init_tables.py

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    text
)
from sqlalchemy.orm import declarative_base
from datetime import datetime
from monitoring.configs.db_config import get_engine

Base = declarative_base()

class DataDrift(Base):
    __tablename__ = "data_drift"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feature_name = Column(String(100))
    drift_score = Column(Float)
    p_value = Column(Float)
    details = Column(JSON)

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100))
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    auc = Column(Float)
    details = Column(JSON)

class TargetDrift(Base):
    __tablename__ = "target_drift"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    drift_score = Column(Float)
    p_value = Column(Float)
    details = Column(JSON)

def create_tables():
    """Create monitoring tables in fraud_monitoring DB."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        print("✅ [SUCCESS] Monitoring tables created successfully.")
    except Exception as e:
        print(f"❌ [ERROR] Table creation failed: {e}")

if __name__ == "__main__":
    create_tables()
