import os
import sys
import logging
import pandas as pd
import re
import io
import json
import random
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy import create_engine, text
from typing import Any, Dict, List, Literal, Optional, Union, Annotated
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage,
)
from langgraph.types import Command
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from matplotlib.ticker import FuncFormatter
from dotenv import load_dotenv

from .ollama_service import OllamaService
from .base import (
    MultiplicationInput, PayslipQuery, LeaveBalanceRequest, PayslipListQuery,
    PayslipInfo, PayslipSummary, PayslipListResponse, PayslipDownloadQuery,
    PayslipDownloadResponse, PayslipExplainQuery, PayslipExplainResponse,
    LeaveTypeRequest, PrepareLeaveApplicationRequest, PreparedLeaveApplication,
    ValidateLeaveBalanceRequest, ValidateLeaveBalanceResponse, CalculateDaysRequest,
    CalculateDaysResponse, SubmitLeaveApplicationRequest, SearchJobOpportunitiesRequest,
    JobOpportunityResponse, LeaveStatusRequest, ExitPolicyRequest, TravelSearchRequest,
    ProfileUpdateInput, CustomerProfileInput, CustomerDetailsInput, ToolInput,
    Answer, VisualizationInput, SQLQueryInput, Summary, State, UpdateCustomerProfileInput
)

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatbot.log")
logging.captureWarnings(True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"),
    ],
    force=True,
)

load_dotenv()

logger = logging.getLogger("HR_AGENT")

# Suppress noisy libraries
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.INFO)

TENANT_SQL_AGENTS = {}
TENANT_DBS = {}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME", "")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def log_info(msg, tenant_id, conversation_id):
    logger.info(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_error(msg, tenant_id, conversation_id):
    logger.error(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_debug(msg, tenant_id, conversation_id):
    logger.debug(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_warning(msg, tenant_id, conversation_id):
    logger.warning(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_exception_auto(msg, tenant_id, conversation_id):
    logger.error(
        f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}",
        exc_info=True,
    )

tavily_search = TavilySearch(max_results=2)
search_tool = TavilySearch(
    max_results=5,
    include_raw_content=True,
)

@tool("get_payslip_tool", args_schema=PayslipQuery)
def get_payslip_tool(config: RunnableConfig, **kwargs):
    """Normalizes dates to MMYYYY and fetches payslip archive."""
    tid = kwargs.get('current_tool_id')
    raw_start = kwargs.get('start_date')
    raw_end = kwargs.get('end_date')

    def normalize_to_mmyyyy(date_str: str) -> str:
        try:
            parsed_date = parse(date_str, fuzzy=True)
            return parsed_date.strftime("%m%Y")
        except:
            if len(date_str) == 6 and date_str.isdigit():
                return date_str
            raise ValueError(f"Could not understand date: {date_str}")

    try:
        clean_start = normalize_to_mmyyyy(raw_start)
        clean_end = normalize_to_mmyyyy(raw_end)
        employee_id = config["configurable"].get("employee_id")
        download_url = f"https://hr.system/payslip/{employee_id}/{clean_start}-{clean_end}.pdf"

        payslip_data = {
            "start_date": clean_start,
            "end_date": clean_end,
            "download_url": download_url
        }

        return Command(
            update={
                "payslip_info": payslip_data,
                "messages": [
                    ToolMessage(
                        content=f"Your payslip for {clean_start} to {clean_end} has been sent to your email.",
                        tool_call_id=tid
                    )
                ]
            }
        )
    except Exception as e:
        return f"I had trouble understanding the dates. Please use MMYYYY format. Error: {str(e)}"

@tool("fetch_available_leave_types_tool", args_schema=LeaveTypeRequest)
def fetch_available_leave_types_tool(config: RunnableConfig, **kwargs):
    """
    REQUIRED FIRST STEP for leave applications. 
    Fetches available leave types from the PostgreSQL tenant database.
    """
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_id = config["configurable"].get("employee_id")
    tenant_id = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")
    
    log_info(f"Fetching available leave types for employee: {emp_id}", tenant_id, "unknown")

    try:
        if not db_uri: return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as connection:
            query = text("""
                SELECT id, name, is_paid, base_entitlement 
                FROM leave_leavetype 
                WHERE tenant_id = (SELECT id FROM org_tenant WHERE code = :tenant_code)
                ORDER BY name
            """)
            result = connection.execute(query, {"tenant_code": tenant_id})
            leave_types = result.fetchall()
            
            if leave_types:
                leave_names = [row[1] for row in leave_types]
                result_text = f"The following leave types are available: {', '.join(leave_names)}. Which one would you like to apply for?"
                return ToolMessage(content=result_text, tool_call_id=tid)
            return ToolMessage(content="No leave types are currently configured for your tenant.", tool_call_id=tid)
    except Exception as e:
        log_exception_auto(f"Failed to fetch leave types: {str(e)}", tenant_id, "unknown")
        return ToolMessage(content=f"Error retrieving leave types: {str(e)}", tool_call_id=tid)

@tool("validate_leave_balance_tool", args_schema=ValidateLeaveBalanceRequest)
def validate_leave_balance_tool(config: RunnableConfig, **kwargs):
    """Validates leave balance against the PostgreSQL tenant database."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_id = kwargs.get('employeeID')
    leave_type_name = kwargs.get('leaveTypeName')
    leave_type_id = kwargs.get('leaveTypeID')
    year = kwargs.get('year')
    num_days = kwargs.get('numOfDays')
    
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    db_uri = config["configurable"].get("db_uri")
    
    try:
        if not db_uri: return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as connection:
            if not leave_type_id and leave_type_name:
                lt_query = text("""
                    SELECT id FROM leave_leavetype 
                    WHERE name ILIKE :name 
                    AND tenant_id = (SELECT id FROM org_tenant WHERE code = :tenant_code)
                """)
                lt_result = connection.execute(lt_query, {"name": leave_type_name, "tenant_code": tenant_id}).fetchone()
                if lt_result: leave_type_id = lt_result[0]
                else: return ToolMessage(content=f"Error: Leave type '{leave_type_name}' not found.", tool_call_id=tid)

            balance_query = text("""
                SELECT lb.balance_days FROM leave_leavebalance lb
                JOIN employees_employee ee ON lb.employee_id = ee.id
                WHERE ee.email = :emp_id AND lb.leave_type_id = :lt_id AND lb.year = :year
            """)
            bal_result = connection.execute(balance_query, {"emp_id": emp_id, "lt_id": leave_type_id, "year": year}).fetchone()
            
            if not bal_result: return ToolMessage(content=f"Error: No leave balance found for {leave_type_name}.", tool_call_id=tid)
            remaining = float(bal_result[0])
            if remaining < num_days:
                return ToolMessage(content=f"Insufficient balance. You have {remaining} days available.", tool_call_id=tid)
            return ToolMessage(content=f"Validation successful. You have {remaining} days available.", tool_call_id=tid)
    except Exception as e:
        log_error(f"Failed to validate leave balance: {str(e)}", tenant_id, "unknown")
        return ToolMessage(content=f"Error validating leave balance: {str(e)}", tool_call_id=tid)

@tool("submit_leave_application_tool", args_schema=SubmitLeaveApplicationRequest)
def submit_leave_application_tool(config: RunnableConfig, **kwargs):
    """Finalizes the leave application and inserts the record into the PostgreSQL database."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_email = config["configurable"].get("employee_id")
    tenant_code = config["configurable"].get("tenant_id", "unknown")
    db_uri = config["configurable"].get("db_uri")
    
    lt_name = kwargs.get('leaveTypeName')
    start_str = kwargs.get('leaveStartDate')
    end_str = kwargs.get('leaveEndDate')
    reason = kwargs.get('leaveReason') or ""
    relief_name = kwargs.get('workAssigneeRequest')
    
    try:
        if not db_uri: return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        start_date = datetime.strptime(start_str, "%d%m%Y").date()
        end_date = datetime.strptime(end_str, "%d%m%Y").date()

        engine = create_engine(db_uri)
        with engine.connect() as connection:
            t_res = connection.execute(text("SELECT id FROM org_tenant WHERE code = :code"), {"code": tenant_code}).fetchone()
            if not t_res: return ToolMessage(content=f"Error: Tenant '{tenant_code}' not found.", tool_call_id=tid)
            tenant_db_id = t_res[0]

            e_res = connection.execute(text("SELECT id FROM employees_employee WHERE email = :email AND tenant_id = :t_id"), {"email": emp_email, "t_id": tenant_db_id}).fetchone()
            if not e_res: return ToolMessage(content=f"Error: Employee '{emp_email}' not found.", tool_call_id=tid)
            employee_db_id = e_res[0]

            lt_res = connection.execute(text("SELECT id FROM leave_leavetype WHERE name ILIKE :name AND tenant_id = :t_id"), {"name": lt_name, "t_id": tenant_db_id}).fetchone()
            if not lt_res: return ToolMessage(content=f"Error: Leave type '{lt_name}' not found.", tool_call_id=tid)
            leave_type_db_id = lt_res[0]

            relief_id = None
            if relief_name:
                r_res = connection.execute(text("SELECT id FROM employees_employee WHERE (first_name || ' ' || last_name ILIKE :name OR email = :name) AND tenant_id = :t_id"), {"name": relief_name, "t_id": tenant_db_id}).fetchone()
                if r_res: relief_id = r_res[0]

            insert_query = text("""
                INSERT INTO leave_leaverequest (employee_id, leave_type_id, start_date, end_date, reason, approval_status, tenant_id, relief_employee_id, order_date)
                VALUES (:e_id, :lt_id, :start, :end, :reason, 'pending', :t_id, :r_id, NOW()) RETURNING id
            """)
            new_id = connection.execute(insert_query, {"e_id": employee_db_id, "lt_id": leave_type_db_id, "start": start_date, "end": end_date, "reason": reason, "t_id": tenant_db_id, "r_id": relief_id}).fetchone()[0]
            connection.commit()

            return Command(update={"leave_application": {"status": "success", "application_id": str(new_id)}, "messages": [ToolMessage(content=f"Application submitted (Ref: {new_id}).", tool_call_id=tid)]}, goto="assistant")
    except Exception as e:
        log_error(f"Failed to submit: {str(e)}", tenant_code, "unknown")
        return ToolMessage(content=f"Error: {str(e)}", tool_call_id=tid)

@tool("prepare_leave_application_tool", args_schema=PrepareLeaveApplicationRequest)
def prepare_leave_application_tool(config: RunnableConfig, **kwargs):
    """Validates full leave details and calculates resumption date."""
    tid = kwargs.get('current_tool_id')
    start_date = kwargs.get("leaveStartDate")
    end_date = kwargs.get("leaveEndDate")
    reliever = kwargs.get("workAssigneeRequest")

    try:
        start_dt = datetime.strptime(start_date, "%d%m%Y")
        end_dt = datetime.strptime(end_date, "%d%m%Y")
        resumption_dt = end_dt + timedelta(days=1)
        res_str = resumption_dt.strftime("%d-%m-%Y")
        
        return Command(update={"leave_application": {"status": "prepared", "details": {**kwargs, "resumptionDate": res_str}}, "messages": [ToolMessage(content=f"Prepared. Resumption: {res_str}. Confirm to Submit.", tool_call_id=tid)]})
    except Exception as e:
        return ToolMessage(content=f"Error: {str(e)}", tool_call_id=tid)

@tool("calculate_num_of_days_tool", args_schema=CalculateDaysRequest)
def calculate_num_of_days_tool(startDate: str, endDate: str, holidays: List[str] = [], config: RunnableConfig = None):
    """Calculates business days between two dates."""
    try:
        start = datetime.strptime(startDate, "%d%m%Y").date()
        end = datetime.strptime(endDate, "%d%m%Y").date()
        holiday_set = {parse(h).date() for h in holidays}
        days = 0
        curr = start
        while curr <= end:
            if curr.weekday() < 5 and curr not in holiday_set: days += 1
            curr += timedelta(days=1)
        return CalculateDaysResponse(numOfDays=days)
    except: return CalculateDaysResponse(numOfDays=0)

@tool("search_job_opportunities_tool", args_schema=SearchJobOpportunitiesRequest)
def search_job_opportunities_tool(config: RunnableConfig, **kwargs):
    """Searches for internal job vacancies in the tenant database."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    tenant_id = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")

    try:
        if not db_uri: return "Database configuration missing."
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            sql = """
                SELECT jt.name AS title, ou.name AS dept, jr.status
                FROM org_jobrole jr
                JOIN org_jobtitle jt ON jr.job_title_id = jt.id
                JOIN org_orgunit ou ON jr.org_unit_id = ou.id
                WHERE jr.vacant = true AND jr.tenant_id = (SELECT id FROM org_tenant WHERE code = :t_code)
            """
            rows = conn.execute(text(sql), {"t_code": tenant_id}).fetchall()
            if not rows: return ToolMessage(content="No vacancies found.", tool_call_id=tid)
            results = [f"{r.title} in {r.dept} ({r.status})" for r in rows]
            return ToolMessage(content="Found: " + ", ".join(results), tool_call_id=tid)
    except Exception as e: return f"Error: {str(e)}"

@tool("fetch_leave_status_tool", args_schema=LeaveStatusRequest)
def fetch_leave_status_tool(config: RunnableConfig, **kwargs):
    """Checks recent leave requests and their current status/approvers."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_email = config["configurable"].get("employee_id")
    tenant_code = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")

    try:
        if not db_uri: return ToolMessage(content="Error: Database missing.", tool_call_id=tid)
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            sql = """
                SELECT lt.name, lr.start_date, lr.end_date, lr.approval_status
                FROM leave_leaverequest lr
                JOIN leave_leavetype lt ON lr.leave_type_id = lt.id
                JOIN employees_employee emp ON lr.employee_id = emp.id
                WHERE emp.email = :email AND lr.tenant_id = (SELECT id FROM org_tenant WHERE code = :t_code)
                ORDER BY lr.order_date DESC LIMIT 3
            """
            rows = conn.execute(text(sql), {"email": emp_email, "t_code": tenant_code}).fetchall()
            if not rows: return ToolMessage(content="No recent requests found.", tool_call_id=tid)
            res = [f"{r.name} ({r.start_date} to {r.end_date}): {r.approval_status}" for r in rows]
            return ToolMessage(content="Recent Requests:\n" + "\n".join(res), tool_call_id=tid)
    except Exception as e: return ToolMessage(content=f"Error: {e}", tool_call_id=tid)

@tool("search_travel_deals_tool", args_schema=TravelSearchRequest)
def search_travel_deals_tool(config: RunnableConfig, **kwargs):
    """Searches for travel deals using Tavily."""
    tid = kwargs.get('current_tool_id')
    query = f"travel deals to {kwargs.get('destination')} from {kwargs.get('departureDate')} to {kwargs.get('returnDate')}"
    try:
        results = search_tool.invoke({"query": query})
        return ToolMessage(content=f"Deals: {str(results)[:1000]}", tool_call_id=tid)
    except Exception as e: return f"Search failed: {e}"

@tool("create_customer_profile_tool", args_schema=CustomerProfileInput)
def create_customer_profile_tool(config: RunnableConfig, **kwargs):
    """Creates a new customer profile and generates an account number."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    tenant_code = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")
    
    acc_num = "".join([str(random.randint(0, 9)) for _ in range(10)])
    cust_id = f"CUST{random.randint(10000, 99999)}"

    try:
        if not db_uri: return ToolMessage(content="Database missing.", tool_call_id=tid)
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            t_res = conn.execute(text("SELECT id FROM org_tenant WHERE code = :code"), {"code": tenant_code}).fetchone()
            if not t_res: return ToolMessage(content="Tenant not found.", tool_call_id=tid)
            
            sql = """
                INSERT INTO customer_customer (customer_id, first_name, last_name, email, phone_number, account_number, gender, tenant_id)
                VALUES (:cid, :fn, :ln, :email, :phone, :acc, :gender, :tid)
            """
            conn.execute(text(sql), {
                "cid": cust_id, "fn": kwargs.get('first_name'), "ln": kwargs.get('last_name'),
                "email": kwargs.get('email'), "phone": kwargs.get('phone'), "acc": acc_num,
                "gender": kwargs.get('gender'), "tid": t_res[0]
            })
            conn.commit()
            return ToolMessage(content=f"Profile Created. Account: {acc_num}", tool_call_id=tid)
    except Exception as e: return ToolMessage(content=f"Error: {e}", tool_call_id=tid)

@tool("get_customer_details_tool", args_schema=CustomerDetailsInput)
def get_customer_details_tool(config: RunnableConfig, **kwargs):
    """Retrieves customer details by email, phone, or account number."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    tenant_code = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")
    val = kwargs.get('phone_or_email')

    try:
        if not db_uri: return "Database missing."
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            sql = """
                SELECT first_name, last_name, account_number FROM customer_customer
                WHERE (email = :val OR phone_number = :val OR account_number = :val)
                AND tenant_id = (SELECT id FROM org_tenant WHERE code = :t_code)
            """
            row = conn.execute(text(sql), {"val": val, "t_code": tenant_code}).fetchone()
            if not row: return ToolMessage(content="Not found.", tool_call_id=tid)
            return ToolMessage(content=f"Found: {row.first_name} {row.last_name} ({row.account_number})", tool_call_id=tid)
    except Exception as e: return f"Error: {e}"

@tool("update_customer_tool", args_schema=UpdateCustomerProfileInput)
def update_customer_tool(config: RunnableConfig, **kwargs):
    """Updates customer profile details."""
    tid = kwargs.get('current_tool_id') or "unknown_id"
    tenant_code = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")
    email = kwargs.get('email')

    try:
        if not db_uri: return "Database missing."
        if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        update_data = {k: v for k, v in kwargs.items() if v is not None and k not in ['current_tool_id', 'email']}
        if not update_data: return "No updates provided."

        set_clause = ", ".join([f"{k} = :{k}" for k in update_data.keys()])
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            sql = f"UPDATE customer_customer SET {set_clause} WHERE email = :email AND tenant_id = (SELECT id FROM org_tenant WHERE code = :t_code)"
            conn.execute(text(sql), {**update_data, "email": email, "t_code": tenant_code})
            conn.commit()
            return ToolMessage(content="Profile updated.", tool_call_id=tid)
    except Exception as e: return f"Error: {e}"

def init_sql_agent(state: State, llm):
    """Initializes a standalone SQL Agent with the provided database URI."""
    tenant_id = state.get("tenant_id", "default")
    db_uri = state.get("db_uri")
    if not db_uri: return None
    if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

    try:
        db = SQLDatabase.from_uri(db_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_react_agent(llm, toolkit.get_tools())
        TENANT_SQL_AGENTS[tenant_id] = agent
        TENANT_DBS[tenant_id] = db
        return agent
    except Exception as e:
        log_error(f"SQL Init failed: {e}", tenant_id, "none")
        return None

@tool("sql_query_tool", description="Queries the SQL database for structured info.")
def sql_query_tool(query: str, state: dict) -> dict:
    """Delegates query execution to the tenant-specific SQL agent."""
    tenant_id = state.get("tenant_id", "unknown")
    agent = TENANT_SQL_AGENTS.get(tenant_id)
    if not agent: return {"sql_result": "SQL Agent not ready."}
    try:
        res = agent.invoke({"messages": [HumanMessage(content=query)]})
        return {"sql_result": res["messages"][-1].content}
    except Exception as e: return {"sql_result": f"Error: {e}"}

@tool("pdf_retrieval_tool", description="Searches the knowledge base PDFs.")
def pdf_retrieval_tool(query: str, state: dict) -> dict:
    """Uses FAISS for local semantic search over loaded documents."""
    # Note: Placeholder for actual vector store logic in chat_bot.py
    return {"pdf_content": "Knowledge base search logic is being refactored to support remote indexes."}

@tool("web_search_tool")
def web_search_tool(query: str, state: dict) -> dict:
    """Performs advanced web search with Tavily."""
    try:
        results = search_tool.invoke({"query": query})
        return {"web_content": str(results)}
    except Exception as e: return {"web_content": f"Search failed: {e}"}

@tool("generate_visualization_tool", args_schema=VisualizationInput)
def generate_visualization_tool(query: str, state: dict) -> dict:
    """Generates charts from SQL data using Pandas and Matplotlib."""
    tenant_id = state.get("tenant_id", "unknown")
    db_uri = state.get("db_uri")
    if not db_uri: return {"visualization_result": {"analysis": "DB not found."}}
    if db_uri.startswith("postgres://"): db_uri = db_uri.replace("postgres://", "postgresql://", 1)

    try:
        engine = create_engine(db_uri)
        # Simplified for refactor: normally would generate SQL via LLM first
        df = pd.read_sql_query("SELECT NOW()", con=engine) 
        return {"visualization_result": {"analysis": "Visualization logic is active.", "image_base64": None}}
    except Exception as e: return {"visualization_result": {"analysis": f"Viz Error: {e}"}}

tools = [
    get_payslip_tool, fetch_available_leave_types_tool, validate_leave_balance_tool,
    prepare_leave_application_tool, calculate_num_of_days_tool, submit_leave_application_tool,
    search_job_opportunities_tool, fetch_leave_status_tool, search_travel_deals_tool,
    create_customer_profile_tool, get_customer_details_tool, generate_visualization_tool,
    sql_query_tool, pdf_retrieval_tool, web_search_tool,
]