import os
from sqlalchemy import create_engine, inspect
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

if __name__ == "__main__":
    inspect_table()
