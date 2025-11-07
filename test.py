from monitoring.configs.db_config import get_engine

engine = get_engine()
print("Connected successfully:", engine)
