import os
import json
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Configuration
LOCAL_URL = "postgresql://postgres:postgres@localhost:5432/hris?sslmode=disable"
REMOTE_URL = "postgresql://postgres:851fa108b40cc528ea77@147.182.194.8:5431/whatsapp-1?sslmode=disable"

# Fix schemes if needed
LOCAL_URL = LOCAL_URL.replace("postgres://", "postgresql://", 1)
REMOTE_URL = REMOTE_URL.replace("postgres://", "postgresql://", 1)

# Default Seed Values
DEFAULT_AGENT_PROMPT = """You are a helpful AI assistant. Your task is to analyze the user's request and decide if a tool is needed."""
DEFAULT_FINAL_ANSWER_PROMPT = """You are Damilola, the AI-powered virtual assistant. Deliver professional customer service and insightful data analysis."""
DEFAULT_TOOL_INTENT_MAP = {
    "leave_management": {"tools": ["fetch_available_leave_types_tool", "prepare_leave_application_tool"], "triggers": ["leave", "vacation"]},
    "data_analysis": {"tools": ["sql_query_tool"], "triggers": ["report", "count", "average"]},
    "visualization": {"tools": ["generate_visualization_tool"], "triggers": ["plot", "chart", "graph"]}
}

def sync_database():
    print("Starting Database Synchronization...")
    
    local_engine = create_engine(LOCAL_URL)
    remote_engine = create_engine(REMOTE_URL)
    
    # 1. Schema Adjustment on Remote
    try:
        with remote_engine.connect() as conn:
            print("  -> Adjusting schema on remote (customer_prompt)...")
            # Using transaction to ensure atomicity
            with conn.begin():
                conn.execute(text("ALTER TABLE customer_prompt ADD COLUMN IF NOT EXISTS agent_prompt TEXT;"))
                conn.execute(text('ALTER TABLE customer_prompt ADD COLUMN IF NOT EXISTS "GLOBAL_FINAL_ANSWER_PROMPT" TEXT;'))
                conn.execute(text('ALTER TABLE customer_prompt ADD COLUMN IF NOT EXISTS "TOOL_INTENT_MAP" JSONB;'))
            print("  [SUCCESS] Schema adjusted.")
    except Exception as e:
        print(f"  [ERROR] Schema adjustment failed: {e}")

    # 2. Wipe Remote Data (Clean Slate)
    tables = [
        "customer_message", 
        "customer_conversation", 
        "customer_tenant_ai", 
        "customer_prompt", 
        "customer_llm",
        "customer_customer",
        "customer_transaction",
        "customer_loanreport",
        "customer_branchperformance"
    ]
    
    try:
        with remote_engine.connect() as conn:
            print("  -> Wiping remote tables...")
            with conn.begin():
                for table in tables:
                    try:
                        conn.execute(text(f"TRUNCATE TABLE {table} CASCADE;"))
                    except Exception as te:
                        print(f"     (Skipped {table}: {te})")
            print("  [SUCCESS] Tables wiped.")
    except Exception as e:
        print(f"  [ERROR] Wipe failed: {e}")

    # 3. Data Transfer
    for table in tables:
        print(f"  -> Migrating data for {table}...")
        try:
            with local_engine.connect() as l_conn:
                local_data = l_conn.execute(text(f"SELECT * FROM {table}")).fetchall()
                if not local_data:
                    print(f"     (No data found locally for {table})")
                    continue
                
                keys = l_conn.execute(text(f"SELECT * FROM {table} LIMIT 0")).keys()
                cols = list(keys)
            
            with remote_engine.connect() as r_conn:
                with r_conn.begin():
                    placeholders = ", ".join([f":{c}" for c in cols])
                    col_names = ", ".join([f'"{c}"' for c in cols])
                    insert_stmt = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
                    
                    for row in local_data:
                        r_conn.execute(text(insert_stmt), dict(zip(cols, row)))
            print(f"  [SUCCESS] Migrated {len(local_data)} rows to {table}.")
        except Exception as e:
            print(f"  [ERROR] Migration failed for {table}: {e}")

    # 4. Seed Standard Prompt
    try:
        with remote_engine.connect() as conn:
            print("  -> Seeding 'standard' prompt behavior...")
            with conn.begin():
                # Check for existing standard prompt
                res = conn.execute(text("SELECT id FROM customer_prompt WHERE name = 'standard'")).fetchone()
                
                params = {
                    "ap": DEFAULT_AGENT_PROMPT,
                    "fp": DEFAULT_FINAL_ANSWER_PROMPT,
                    "tm": json.dumps(DEFAULT_TOOL_INTENT_MAP)
                }
                
                if res:
                    conn.execute(text("""
                        UPDATE customer_prompt 
                        SET agent_prompt = :ap, 
                            "GLOBAL_FINAL_ANSWER_PROMPT" = :fp, 
                            "TOOL_INTENT_MAP" = :tm::jsonb
                        WHERE name = 'standard'
                    """), params)
                else:
                    conn.execute(text("""
                        INSERT INTO customer_prompt (name, agent_prompt, "GLOBAL_FINAL_ANSWER_PROMPT", "TOOL_INTENT_MAP")
                        VALUES ('standard', :ap, :fp, :tm::jsonb)
                    """), params)
            print("  [SUCCESS] Standard prompt seeded.")
    except Exception as e:
        print(f"  [ERROR] Seeding failed: {e}")

    print("\nDatabase Synchronization Process Finished.")

if __name__ == "__main__":
    sync_database()
