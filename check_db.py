import sqlite3
import os

db_path = r"c:\Users\kumma\OneDrive\Desktop\Downloads\disability-rights-guide\backend\disability_rights.db"
if not os.path.exists(db_path):
    print(f"File not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"Tables: {tables}")

for table_name in tables:
    table_name = table_name[0]
    print(f"\nSchema for {table_name}:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    print(cursor.fetchall())
    
    # Check if there's any data in it
    cursor.execute(f"SELECT count(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")
    
    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        print(f"Sample data: {cursor.fetchall()}")

conn.close()
