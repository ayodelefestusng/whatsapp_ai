import os
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

LOCAL_URL = "postgresql://postgres:postgres@localhost:5432/hris?sslmode=disable"
LOCAL_URL = LOCAL_URL.replace("postgres://", "postgresql://", 1)

def inspect_table():
    engine = create_engine(LOCAL_URL)
    inspector = inspect(engine)
    columns = inspector.get_columns('customer_prompt')
    print("Columns in customer_prompt:")
    for col in columns:
        print(f"  - {col['name']} ({col['type']})")

def show_records():
    engine = create_engine(LOCAL_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM customer_prompt LIMIT 10"))
        print("\nRecords in customer_prompt:")
        for row in result:
            print(row)

if __name__ == "__main__":
    inspect_table()
    show_records()