# monitoring/test_mysql_connection.py
from sqlalchemy import create_engine, text

def test_mysql_connection_and_table():
    """Test MySQL connection and fix the table schema"""
    
    # Test connection
    try:
        engine = create_engine("mysql+pymysql://root:@localhost:3306/fraud_monitoring")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ MySQL connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # # Check if table exists and fix schema
    # try:
    #     with engine.begin() as conn:
    #         # Check if table exists
    #         result = conn.execute(text("""
    #             SELECT TABLE_NAME 
    #             FROM INFORMATION_SCHEMA.TABLES 
    #             WHERE TABLE_SCHEMA = 'fraud_monitoring' 
    #             AND TABLE_NAME = 'model_monitoring_metrics'
    #         """))
    #         table_exists = result.fetchone() is not None
            
    #         if table_exists:
    #             print("✅ Table 'model_monitoring_metrics' exists")
                
    #             # Check current schema
    #             result = conn.execute(text("""
    #                 DESCRIBE model_monitoring_metrics
    #             """))
    #             print("\n📋 Current table schema:")
    #             for row in result:
    #                 print(f"  {row}")
                    
    #             # Drop the old table (be careful!)
    #             drop_choice = input("\n❓ Drop and recreate table? (y/n): ")
    #             if drop_choice.lower() == 'y':
    #                 conn.execute(text("DROP TABLE model_monitoring_metrics"))
    #                 print("✅ Old table dropped")
    #             else:
    #                 print("⚠️  Keeping existing table")
    #                 return
    #         else:
    #             print("📝 Table doesn't exist, creating new one...")
            
    #         # Create table with corrected column names
    #         conn.execute(text("""
    #             CREATE TABLE model_monitoring_metrics (
    #                 id INT AUTO_INCREMENT PRIMARY KEY,
    #                 timestamp DATETIME,
    #                 precision_score FLOAT,
    #                 recall_score FLOAT,
    #                 f1_score FLOAT,
    #                 auc_score FLOAT,
    #                 drift_detected BOOLEAN
    #             )
    #         """))
    #         print("✅ Table created successfully with corrected schema!")
            
    # except Exception as e:
    #     print(f"❌ Table operation failed: {e}")

if __name__ == "__main__":
    test_mysql_connection_and_table()