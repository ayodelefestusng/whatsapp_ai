import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Local DB Configuration
LOCAL_URL = "postgresql://postgres:postgres@localhost:5432/hris?sslmode=disable"
LOCAL_URL = LOCAL_URL.replace("postgres://", "postgresql://", 1)

# Default Seed Values
DEFAULT_AGENT_PROMPT = """You are a helpful AI assistant. Your task is to analyze the user's request and decide if a tool is needed."""
DEFAULT_FINAL_ANSWER_PROMPT = """You are Damilola, the AI-powered virtual assistant. Deliver professional customer service and insightful data analysis."""
DEFAULT_TOOL_INTENT_MAP = {
    "leave_management": {"tools": ["fetch_available_leave_types_tool", "prepare_leave_application_tool"], "triggers": ["leave", "vacation"]},
    "data_analysis": {"tools": ["sql_query_tool"], "triggers": ["report", "count", "average"]},
    "visualization": {"tools": ["generate_visualization_tool"], "triggers": ["plot", "chart", "graph"]}
}

def seed_local():
    print("Seeding local database prompt fields...")
    engine = create_engine(LOCAL_URL)
    
    params = {
        "ap": DEFAULT_AGENT_PROMPT,
        "fp": DEFAULT_FINAL_ANSWER_PROMPT,
        "tm": json.dumps(DEFAULT_TOOL_INTENT_MAP)
    }
    
    try:
        with engine.connect() as conn:
            with conn.begin():
                # Check if 'standard' prompt exists
                res = conn.execute(text("SELECT id FROM customer_prompt WHERE name = 'standard'")).fetchone()
                
                if res:
                    print(f"  -> Updating existing 'standard' prompt (ID: {res[0]})")
                    conn.execute(text("""
                        UPDATE customer_prompt 
                        SET agent_prompt = :ap, 
                            "GLOBAL_FINAL_ANSWER_PROMPT" = :fp, 
                            "TOOL_INTENT_MAP" = :tm::jsonb
                        WHERE name = 'standard'
                    """), params)
                else:
                    print("  -> Creating new 'standard' prompt entry")
                    conn.execute(text("""
                        INSERT INTO customer_prompt (name, agent_prompt, "GLOBAL_FINAL_ANSWER_PROMPT", "TOOL_INTENT_MAP")
                        VALUES ('standard', :ap, :fp, :tm::jsonb)
                    """), params)
            print("  [SUCCESS] Local database seeded.")
    except Exception as e:
        print(f"  [ERROR] Seeding failed: {e}")

if __name__ == "__main__":
    seed_local()
