from logger_utils import log_info, log_error, log_debug, log_warning, logger
from llm_handler import get_model
import os
import sys
import requests
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy import create_engine
from typing import Any, Dict, List, Literal, Optional, Union, Annotated
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit



# from myproject_revisit.org.management.commands import state

from ollama_service import OllamaService
from base import (MultiplicationInput,
PayslipQuery,LeaveBalanceRequest,PayslipListQuery,PayslipInfo,PayslipSummary,PayslipListResponse,
PayslipDownloadQuery,PayslipDownloadResponse,PayslipExplainQuery,PayslipExplainResponse,LeaveTypeRequest,
PrepareLeaveApplicationRequest,PreparedLeaveApplication,ValidateLeaveBalanceRequest,
ValidateLeaveBalanceResponse,CalculateDaysRequest,CalculateDaysResponse,SubmitLeaveApplicationRequest,SearchJobOpportunitiesRequest,
JobOpportunityResponse,LeaveStatusRequest,ExitPolicyRequest,TravelSearchRequest,ProfileUpdateInput,
CustomerProfileInput,CustomerDetailsInput,ToolInput,Answer,VisualizationInput,SQLQueryInput,Summary,State,UpdateCustomerProfileInput
           )
# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Logging and configuration - moved to logger_utils and shared modules

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


@tool("get_payslip_tool", args_schema=PayslipQuery, description="Useful for fetching payslip archives. Input dates can be in various formats (e.g., 'Jan 2025', '01/25', '2025-01', '012025'). Outputs a download URL for the specified period.")
def get_payslip_tool(config: RunnableConfig, **kwargs):
    """Normalizes dates to MMYYYY and fetches payslip archive."""
    
    tid = kwargs.get('current_tool_id')
    raw_start = kwargs.get('start_date')
    raw_end = kwargs.get('end_date')
    

    def normalize_to_mmyyyy(date_str: str) -> str:
        try:
            
            # Smart parsing: handles "Jan 2025", "01/25", "2025-01", etc.
            parsed_date = parse(date_str, fuzzy=True)
            return parsed_date.strftime("%m%Y")
        except:
            # Fallback for purely numeric strings like "012025"
            if len(date_str) == 6 and date_str.isdigit():
                return date_str
            raise ValueError(f"Could not understand date: {date_str}")

    try:
        clean_start = normalize_to_mmyyyy(raw_start)
        clean_end = normalize_to_mmyyyy(raw_end)
        
        employee_id = config["configurable"].get("employee_id")
        download_url = f"https://hr.system/payslip/{employee_id}/{clean_start}-{clean_end}.pdf"

        # The data structure you requested for PayslipInfo
        payslip_data = {
            "start_date": clean_start, # Now guaranteed MMYYYY
            "end_date": clean_end,     # Now guaranteed MMYYYY
            "download_url": download_url
        }

        return Command(
            update={
                "payslip_info": payslip_data, # Added as child to State
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
    # 1. Get the ID we injected in tool_node for the response message
    tid = kwargs.get('current_tool_id')
    if not tid:
        tid = "unknown_id" 
    
    emp_id = config["configurable"].get("employee_id")
    tenant_id = config["configurable"].get("tenant_id")
    db_uri = config["configurable"].get("db_uri")
    
    log_info(f"Fetching available leave types for employee: {emp_id} from tenant: {tenant_id}", tenant_id, "unknown")

    try:
        # Validate db_uri is available
        if not db_uri:
            log_error(f"No database URI available for tenant {tenant_id}", tenant_id, "unknown")
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        # Ensure PostgreSQL URI uses postgresql:// prefix
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        # Create engine and execute query
        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # Query the leave_leavetype table for available leave types for this tenant
                query = text("""
                    SELECT id, name, is_paid, base_entitlement 
                    FROM leave_leavetype 
                    WHERE tenant_id = (
                        SELECT id FROM org_tenant WHERE code = :tenant_code
                    )
                    ORDER BY name
                """)
                
                result = connection.execute(query, {"tenant_code": tenant_id})
                leave_types = result.fetchall()
                
                if leave_types:
                    # Extract leave type names
                    leave_names = [row[1] for row in leave_types]  # row[1] is the name column
                    result_text = f"The following leave types are available: {', '.join(leave_names)}. Which one would you like to apply for?"
                    log_info(f"The result text from the fetch_available_leave_types_tool: {result_text}", tenant_id, "unknown")
                    return ToolMessage(content=result_text, tool_call_id=tid)
                else:
                    return ToolMessage(content="No leave types are currently configured for your tenant.", tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
        log_info(f"Failed to fetch leave types from database: {str(e)}", tenant_id, "unknown")
        return ToolMessage(content=f"Error retrieving leave types: {str(e)}", tool_call_id=tid)



@tool("validate_leave_balance_tool", args_schema=ValidateLeaveBalanceRequest)
def validate_leave_balance_tool(config: RunnableConfig, **kwargs):
    """
    Validates leave balance against the PostgreSQL tenant database.
    If numOfDays is 0, it calculates it using the dates provided in the workflow state.
    """
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_id = kwargs.get('employeeID')
    leave_type_name = kwargs.get('leaveTypeName')
    leave_type_id = kwargs.get('leaveTypeID')
    year = kwargs.get('year')
    num_days = kwargs.get('numOfDays')
    
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    db_uri = config["configurable"].get("db_uri")
    
    log_info(f"Validating leave balance for employee: {emp_id} (Type: {leave_type_name})", tenant_id, conversation_id)

    try:
        if not db_uri:
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve leave_type_id if only name is provided
                if not leave_type_id and leave_type_name:
                    lt_query = text("""
                        SELECT id FROM leave_leavetype 
                        WHERE name ILIKE :name 
                        AND tenant_id = (SELECT id FROM org_tenant WHERE code = :tenant_code)
                    """)
                    lt_result = connection.execute(lt_query, {"name": leave_type_name, "tenant_code": tenant_id}).fetchone()
                    if lt_result:
                        leave_type_id = lt_result[0]
                    else:
                        return ToolMessage(content=f"Error: Leave type '{leave_type_name}' not found.", tool_call_id=tid)

                # 2. Fetch balance
                # Query leave_leavebalance table
#                 balance_query = text("""
#     SELECT lb.balance_days, lb.total_earned, lb.used 
#     FROM leave_leavebalance lb
#     JOIN employees_employee ee ON lb.employee_id = ee.id
#     JOIN auth_user au ON ee.user_id = au.id
#     WHERE au.email = :emp_id 
#     AND lb.leave_type_id = :lt_id 
#     AND lb.year = :year
# """)
                
                
                balance_query = text("""
                    SELECT lb.balance_days, lb.total_earned, lb.used 
                    FROM leave_leavebalance lb
                    JOIN employees_employee ee ON lb.employee_id = ee.id
                    WHERE ee.employee_email = :emp_id 
                    AND lb.leave_type_id = :lt_id 
                    AND lb.year = :year
                """)
                
                try:
                    bal_result = connection.execute(balance_query, {
                        "emp_id": emp_id, 
                        "lt_id": leave_type_id, 
                        "year": year
                    }).fetchone()
                except Exception as e:
                    # HERE IS WHERE YOU PUT THE LOGGING LOGIC
                    logger.error(f"DB Error validating balance for {emp_id}: {str(e)}")
                    return ToolMessage(
                        content="We encountered an issue checking your leave balance. Please try again or contact HR support.", 
                        tool_call_id=tid
                    )
                
                if not bal_result:
                    return ToolMessage(content=f"Error: No leave balance record found for {leave_type_name} in {year}.", tool_call_id=tid)
                
                remaining = float(bal_result[0])
                
                if remaining < num_days:
                    result_text = f"Insufficient balance. You requested {num_days} days but only have {remaining} days available for {leave_type_name}."
                else:
                    result_text = f"Validation successful. You have {remaining} days available for {leave_type_name}."
                
                return ToolMessage(content=result_text, tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
        log_error(f"Failed to validate leave balance: {str(e)}", tenant_id, conversation_id)
        return ToolMessage(content=f"Error validating leave balance: {str(e)}", tool_call_id=tid)


@tool("submit_leave_application_tool", args_schema=SubmitLeaveApplicationRequest)
def submit_leave_application_tool(config: RunnableConfig, **kwargs):
    """
    Finalizes the leave application and inserts the record into the PostgreSQL database.
    Resolves employee, leave type, and relief employee IDs from the database.
    """
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_email = config["configurable"].get("employee_id")
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    db_uri = config["configurable"].get("db_uri")
    
    # Extract fields from kwargs
    lt_name = kwargs.get('leaveTypeName')
    start_str = kwargs.get('leaveStartDate')
    end_str = kwargs.get('leaveEndDate')
    reason = kwargs.get('leaveReason') or ""
    relief_name = kwargs.get('workAssigneeRequest')
    
    log_info(f"Submitting leave application for: {emp_email} (Type: {lt_name})", tenant_id, conversation_id)

    try:
        if not db_uri:
            log_error("Database configuration missing.", tenant_id, conversation_id)
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from datetime import datetime
        # Parse dates DDMMYYYY
        try:
            start_date = datetime.strptime(start_str, "%d%m%Y").date()
            end_date = datetime.strptime(end_str, "%d%m%Y").date()
        except Exception as e:
            return ToolMessage(content=f"Error parsing dates: {str(e)}. Use DDMMYYYY format.", tool_call_id=tid)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve Tenant ID
                t_query = text("SELECT id FROM org_tenant WHERE code = :code")
                t_res = connection.execute(t_query, {"code": tenant_code}).fetchone()
                if not t_res:
                    return ToolMessage(content=f"Error: Tenant '{tenant_code}' not found.", tool_call_id=tid)
                tenant_id = t_res[0]

                # 2. Resolve Main Employee ID
                e_query = text("SELECT id FROM employees_employee WHERE employee_email = :email AND tenant_id = :t_id")
                e_res = connection.execute(e_query, {"email": emp_email, "t_id": tenant_id}).fetchone()
                if not e_res:
                    return ToolMessage(content=f"Error: Employee profiling for '{emp_email}' not found.", tool_call_id=tid)
                employee_id = e_res[0]

                # 3. Resolve Leave Type ID
                lt_query = text("SELECT id FROM leave_leavetype WHERE name ILIKE :name AND tenant_id = :t_id")
                lt_res = connection.execute(lt_query, {"name": lt_name, "t_id": tenant_id}).fetchone()
                if not lt_res:
                    return ToolMessage(content=f"Error: Leave type '{lt_name}' not found.", tool_call_id=tid)
                leave_type_id = lt_res[0]

                # 4. Resolve Relief Employee ID (Optional)
                relief_id = None
                if relief_name and relief_name.strip():
                    r_query = text("""
                        SELECT id FROM employees_employee 
                        WHERE (first_name || ' ' || last_name ILIKE :name OR employee_email = :name)
                        AND tenant_id = :t_id
                    """)
                    r_res = connection.execute(r_query, {"name": relief_name, "t_id": tenant_id}).fetchone()
                    if r_res:
                        relief_id = r_res[0]

                # 5. Insert Leave Request
                # Using columns from LeaveRequest model: employee_id, leave_type_id, start_date, end_date, reason, approval_status, tenant_id, relief_employee_id
                insert_query = text("""
                    INSERT INTO leave_leaverequest (
                        employee_id, leave_type_id, start_date, end_date, 
                        reason, approval_status, tenant_id, relief_employee_id,
                        order_date
                    ) VALUES (
                        :e_id, :lt_id, :start, :end, 
                        :reason, 'pending', :t_id, :r_id,
                        NOW()
                    ) RETURNING id
                """)
                
                result = connection.execute(insert_query, {
                    "e_id": employee_id,
                    "lt_id": leave_type_id,
                    "start": start_date,
                    "end": end_date,
                    "reason": reason,
                    "t_id": tenant_id,
                    "r_id": relief_id
                })
                new_id = result.fetchone()[0]
                connection.commit()

                success_msg = f"Leave application submitted successfully! Reference ID: {new_id}."
                
                return Command(
                    update={
                        "leave_application": {
                            "status": "success",
                            "application_id": str(new_id)
                        },
                        "messages": [
                            ToolMessage(content=success_msg, tool_call_id=tid)
                        ]
                    },
                    goto="assistant"
                )
        finally:
            engine.dispose()

    except Exception as e:
        log_error(f"Failed to submit leave application: {str(e)}", tenant_id, conversation_id)
        return ToolMessage(content=f"Error submitting application: {str(e)}", tool_call_id=tid)


@tool("prepare_leave_application_tool", args_schema=PrepareLeaveApplicationRequest)
def prepare_leave_application_tool(config: RunnableConfig, **kwargs):
    """Validates full leave details including contact info and year selection."""
    
    tid = kwargs.get('current_tool_id')
    
    # Extract fields from Assistant's tool call
    start_date = kwargs.get("leaveStartDate")
    end_date = kwargs.get("leaveEndDate")
    leave_year = kwargs.get("leaveYear")
    reliever = kwargs.get("workAssigneeRequest")
    address = kwargs.get("addressWhileOnLeave")
    contact = kwargs.get("contactNoWhileOnLeave")
    email = kwargs.get("emailWhileOnLeave")

    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    
    log_info(f"Preparing application for Year {leave_year}. Reliever: {reliever}", tenant_id, conversation_id)

    # 1. Date Format Validation (DDMMYYYY)
    date_fmt = "%d%m%Y"
    try:
        if not start_date or not end_date:
            raise ValueError("Start and End dates are mandatory.")
            
        start_dt_obj = datetime.strptime(start_date, date_fmt)
        end_dt_obj = datetime.strptime(end_date, date_fmt)
        
        # 2. Dynamic Resumption Calculation (End Date + 1)
        resumption_dt_obj = end_dt_obj + timedelta(days=1)
        resumption_str = resumption_dt_obj.strftime("%d-%m-%Y")
        
    except ValueError as e:
        return ToolMessage(
            content=f"Error: {str(e)}. Please ensure dates are in DDMMYYYY format (e.g., 22122025).",
            tool_call_id=tid
        )

    log_info(f"Calculated resumption date as {resumption_str}", tenant_id, conversation_id)

    # 3. Update State
    return Command(
        update={
            "leave_application": {
                "status": "prepared",
                "details": {
                    **kwargs, 
                    "resumptionDate": resumption_str
                }
            },
            "messages": [
                ToolMessage(
                    content=(
                        f"I've prepared your application:\n"
                        f"* Resumption: {resumption_str}\n"
                        f"* Reliever: {reliever}\n"
                        f"* Address: {address}\n"
                        "Please confirm to **Submit**."
                    ),
                    tool_call_id=tid
                )
            ]
        }
    )


@tool("calculate_num_of_days_tool", args_schema=CalculateDaysRequest)
def calculate_num_of_days_tool(startDate: str, endDate: str, holidays: List[str] = [], config: RunnableConfig = None):
    """
    Calculates the actual number of leave days by skipping weekends and 
    a provided list of public holidays.
    """
    tenant_id = config.get("configurable", {}).get("tenant_id", "unknown") if config else "unknown"
    conversation_id = config.get("configurable", {}).get("conversation_id", "unknown") if config else "unknown"
    log_info(f"Calculating business days between {startDate} and {endDate}", tenant_id, conversation_id)
    
    try:
        # 1. Parse string dates to date objects

        # To this:
        start = datetime.strptime(startDate, "%d%m%Y").date()
        end = datetime.strptime(endDate, "%d%m%Y").date()
        
        # 2. Convert holiday strings to a set of date objects for O(1) lookup
        holiday_set = {datetime.fromisoformat(h.replace("Z", "")).date() for h in holidays}
        
        total_days = 0
        current_day = start
        
        # 3. Iterate through the range
        while current_day <= end:
            # Check if it's a weekday (Monday=0, Friday=4) AND not a holiday
            if current_day.weekday() < 5 and current_day not in holiday_set:
                total_days += 1
            else:
                log_reason = "Weekend" if current_day.weekday() >= 5 else "Public Holiday"
                logger.warning(f"Skipping {current_day}: {log_reason}", config)
                
            current_day += timedelta(days=1)

        logger.debug(f"Final calculation: {total_days} business days", config)

        # 4. Return structured response
        return CalculateDaysResponse(numOfDays=total_days)

    except Exception as e:
        log_debug(f"Error calculating leave days: {e}", tenant_id, conversation_id)
        # Fallback to a standard diff if parsing fails, or return 0
        return CalculateDaysResponse(numOfDays=0)


# 2. Update the Tool Function
@tool("search_job_opportunities_tool", args_schema=SearchJobOpportunitiesRequest)
def search_job_opportunities_tool(config: RunnableConfig, **kwargs):
    """Searches for internal job opportunities by querying the tenant's PostgreSQL database.

    This tool uses the `org_jobrole` table and related joins. Only roles marked `vacant = true`
    are considered. Additional filters (department, jobType, location, jobRoleType) map to
    corresponding columns/joined tables where possible.
    """
    tid = kwargs.get('current_tool_id')
    if not tid:
        tid = "unknown_id"
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    db_uri = config["configurable"].get("db_uri")

    log_info(f"Searching jobs for tenant {tenant_id} using db_uri {db_uri}", tenant_id, conversation_id)

    try:
        if not db_uri:
            raise ValueError("No db_uri provided for tenant")
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        try:
            with engine.connect() as conn:
                base_sql = [
                    "SELECT jr.id, jt.name AS job_title, ou.name AS org_unit, jr.vacant, jr.role_type, jr.status",
                    "FROM org_jobrole jr",
                    "LEFT JOIN org_jobtitle jt ON jr.job_title_id = jt.id",
                    "LEFT JOIN org_orgunit ou ON jr.org_unit_id = ou.id",
                    "WHERE jr.vacant = true",
                    "AND jr.tenant_id = (SELECT id FROM org_tenant WHERE code = :tenant_code)"
                ]
                params = {"tenant_code": tenant_id}

                # optional filters
                if kwargs.get("department"):
                    base_sql.append("AND ou.name ILIKE :department")
                    params["department"] = f"%{kwargs['department']}%"
                if kwargs.get("jobType"):
                    base_sql.append("AND jr.status = :jobType")
                    params["jobType"] = kwargs.get("jobType")
                if kwargs.get("location"):
                    base_sql.append("AND ou.name ILIKE :location")
                    params["location"] = f"%{kwargs['location']}%"
                if kwargs.get("jobRoleType"):
                    base_sql.append("AND jr.role_type = :jobRoleType")
                    params["jobRoleType"] = kwargs.get("jobRoleType")

                base_sql.append("ORDER BY jt.name")
                final_query = text("\n".join(base_sql))

                result = conn.execute(final_query, params)
                rows = result.fetchall()
                log_info(f"The rows from the search_job_opportunities_tool: {rows}", tenant_id, conversation_id)

                jobs_found = []
                for row in rows:
                    jobs_found.append({
                        "Role": row.job_title,
                        "OrgUnit": row.org_unit,
                        "Type": row.status,
                        "RoleType": row.role_type,
                    })

                if not jobs_found:
                    result_text = "No open internal positions match those criteria at the moment."
                    log_info(f"The result text from the search_job_opportunities_tool: {result_text}", tenant_id, conversation_id)
                else:
                    result_text = f"Found {len(jobs_found)} opportunities: {jobs_found}"
                    log_info(f"The result text from the search_job_opportunities_tool: {result_text}", tenant_id, conversation_id)
                return ToolMessage(content=result_text, tool_call_id=tid)
        finally:
            engine.dispose()
    except Exception as e:
        log_error(f"Job search failed: {str(e)}", tenant_id, conversation_id)
        return f"Error searching jobs: {str(e)}"
@tool("fetch_leave_status_tool", args_schema=LeaveStatusRequest)
def fetch_leave_status_tool(config: RunnableConfig, **kwargs):
    """Checks the current status of the employee's leave requests and identifies the pending approver and latest comments."""
    
    tid = kwargs.get('current_tool_id') or "unknown_id"
    emp_email = config["configurable"].get("employee_id")
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    db_uri = config["configurable"].get("db_uri")

    log_info(f"Fetching leave status for: {emp_email}", tenant_id, conversation_id)

    try:
        if not db_uri:
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve Tenant ID
                t_query = text("SELECT id FROM org_tenant WHERE code = :code")
                tenant_id = connection.execute(t_query, {"code": tenant_code}).fetchone()
                if not tenant_id:
                    return ToolMessage(content=f"Error: Tenant '{tenant_code}' not found.", tool_call_id=tid)
                tenant_id = tenant_id[0]

                # 2. Query Leave Requests with Workflow data
                # We need to find the ContentType ID for leave.leaverequest
                ct_query = text("SELECT id FROM django_content_type WHERE app_label = 'leave' AND model = 'leaverequest'")
                ct_res = connection.execute(ct_query).fetchone()
                ct_id = ct_res[0] if ct_res else None

                sql = """
                    SELECT 
                        lr.id, 
                        lt.name as leave_type, 
                        lr.start_date, 
                        lr.end_date, 
                        lr.approval_status,
                        wi.id as workflow_id,
                        (
                            SELECT wa.comment 
                            FROM workflow_workflowaction wa 
                            WHERE wa.instance_id = wi.id 
                            ORDER BY wa.created_at DESC LIMIT 1
                        ) as last_comment,
                        (
                            SELECT STRING_AGG(ee.first_name || ' ' || ee.last_name, ', ')
                            FROM workflow_workflowinstance_current_approvers wica
                            JOIN employees_employee ee ON wica.employee_id = ee.id
                            WHERE wica.workflowinstance_id = wi.id
                        ) as current_approvers
                    FROM leave_leaverequest lr
                    JOIN leave_leavetype lt ON lr.leave_type_id = lt.id
                    JOIN employees_employee emp ON lr.employee_id = emp.id
                    LEFT JOIN workflow_workflowinstance wi ON lr.id = wi.object_id AND wi.content_type_id = :ct_id
                    WHERE emp.employee_email = :email AND lr.tenant_id = :t_id
                    ORDER BY lr.order_date DESC
                    LIMIT 5
                """
                
                result = connection.execute(text(sql), {"email": emp_email, "t_id": tenant_id, "ct_id": ct_id})
                rows = result.fetchall()
                
                if not rows:
                    return ToolMessage(content="You have no recent leave requests.", tool_call_id=tid)
                
                reports = []
                for row in rows:
                    status_line = f"**{row.leave_type}** ({row.start_date} to {row.end_date}): **{row.approval_status.upper()}**"
                    if row.current_approvers:
                        status_line += f"\n   - Pending with: {row.current_approvers}"
                    if row.last_comment:
                        status_line += f"\n   - Last Comment: \"{row.last_comment}\""
                    reports.append(status_line)
                
                result_text = "Here is the status of your recent leave requests:\n\n" + "\n\n".join(reports)
                return ToolMessage(content=result_text, tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
        log_error(f"Failed to fetch leave status: {str(e)}", tenant_id, conversation_id)
        return ToolMessage(content=f"Error fetching leave status: {str(e)}", tool_call_id=tid)



@tool("search_travel_deals_tool", args_schema=TravelSearchRequest)
def search_travel_deals_tool(config: RunnableConfig, **kwargs):
    """Uses Tavily to search for the best flight and hotel deals for a vacation."""
    
    tid = kwargs.get('current_tool_id')
    dest = kwargs.get('destination')
    start = kwargs.get('departureDate')
    end = kwargs.get('returnDate')

    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    
    # Construct a high-quality search query for Tavily
    query = f"cheapest flights and top rated hotels in {dest} from {start} to {end} for vacation"
    
    log_info(f"Tavily searching travel deals: {query}", tenant_id, conversation_id)

    try:
        # Execute Tavily Search
        search_results = search_tool.invoke({"query": query})
        
        # Format the output for the AI Assistant
        formatted_results = []
        for res in search_results:
            formatted_results.append({
                "source": res.get("url"),
                "content": res.get("content")[:500] # Snippet for context
            })

        return ToolMessage(
            content=f"Found these travel options in {dest}:\n{str(formatted_results)}",
            tool_call_id=tid
        )
    except Exception as e:
        return f"Travel search failed: {str(e)}"   



@tool("create_customer_profile_tool", args_schema=CustomerProfileInput)
def create_customer_profile_tool(config: RunnableConfig, **kwargs):
    """
    Creates a new customer profile in the PostgreSQL database.
    Generates a unique 10-digit account number and returns it.
    """
    tid = kwargs.get('current_tool_id') or "unknown_id"
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    db_uri = config["configurable"].get("db_uri")
    
    first_name = kwargs.get('first_name')
    last_name = kwargs.get('last_name')
    email = kwargs.get('email')
    phone = kwargs.get('phone')
    gender = kwargs.get('gender')
    dob_str = kwargs.get('date_of_birth')
    occupation = kwargs.get('occupation') or "Not Specified"
    nationality = kwargs.get('nationality') or "Nigeria"
    
    log_info(f"Creating customer profile for: {first_name} {last_name}", tenant_id, conversation_id)

    try:
        if not db_uri:
            log_error("Database configuration missing.", tenant_id, conversation_id)
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from datetime import datetime
        try:
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        except:
            return ToolMessage(content="Error: Invalid date format for date_of_birth. Use YYYY-MM-DD.", tool_call_id=tid)

        import random
        # Generate 10-digit account number
        account_num = "".join([str(random.randint(0, 9)) for _ in range(10)])
        # Generate customer ID
        cust_id = f"CUST{random.randint(10000, 99999)}"

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve Tenant ID
                t_query = text("SELECT id FROM org_tenant WHERE code = :code")
                tenant_id = connection.execute(t_query, {"code": tenant_code}).fetchone()
                if not tenant_id:
                    return ToolMessage(content=f"Error: Tenant '{tenant_code}' not found.", tool_call_id=tid)
                tenant_id = tenant_id[0]

                # 2. Insert Customer
                sql = """
                    INSERT INTO customer_customer (
                        customer_id, first_name, last_name, email, phone_number,
                        account_number, gender, nationality, occupation, date_of_birth,
                        tenant_id
                    ) VALUES (
                        :cid, :fn, :ln, :email, :phone,
                        :acc, :gender, :nat, :occ, :dob,
                        :tid
                    ) RETURNING id
                """
                
                result = connection.execute(text(sql), {
                    "cid": cust_id, "fn": first_name, "ln": last_name, "email": email, "phone": phone,
                    "acc": account_num, "gender": gender, "nat": nationality, "occ": occupation, "dob": dob,
                    "tid": tenant_id
                })
                connection.commit()
                
                msg = f"Successfully created customer profile for {first_name} {last_name}.\nAccount Number: **{account_num}**\nCustomer ID: {cust_id}"
                return ToolMessage(content=msg, tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
        log_error(f"Failed to create customer profile: {str(e)}", tenant_id, conversation_id)
        return ToolMessage(content=f"Error creating customer profile: {str(e)}", tool_call_id=tid)


def init_sql_agent(state: State, llm):
    """Initialize SQLDatabase and SQL Agent using db_uri from state."""
    tenant_id = state.get("tenant_id", "default")
    conversation_id = state.get("conversation_id", "unknown")
    db_uri = state.get("db_uri")  # <-- fetch db_uri from state
    if db_uri == "ayuladb":
        db_uri = DB_URI # Fixed legacy reference
    if db_uri and db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)

    log_info(f"The ule of sql {db_uri} ", tenant_id, conversation_id)

    if not db_uri:
        log_error(f"[{tenant_id}] No db_uri found in state.", tenant_id, conversation_id)
        return None

    try:
        # Import psycopg2 if using PostgreSQL to ensure dialect is available
        if "postgresql" in db_uri:
            try:
                import psycopg2
            except ImportError:
                log_error(f"[{tenant_id}] psycopg2 not installed. Cannot connect to PostgreSQL. {db_uri}", tenant_id, conversation_id)
                return None
        
        db = SQLDatabase.from_uri(db_uri)
        log_info(f"[{tenant_id}] SQLDatabase connected to RE {db_uri} successfully.",tenant_id, conversation_id,)
        # --- Log dialect ---
        log_info(f"[{tenant_id}] Dialect RE: {db.dialect}", tenant_id, conversation_id)

        # --- Log usable tables ---
        try:
            usable_tables = db.get_usable_table_names()
            log_info(f"[{tenant_id}] Available tables RE: {usable_tables}",tenant_id,conversation_id,)
        
        except Exception as e:
            log_warning( f"[{tenant_id}] Could not retrieve usable tables: {e}",tenant_id,conversation_id,)
            usable_tables = []

        # --- Initialize toolkit and agent ---

        # Use a LangChain-compatible OllamaService for the SQL toolkit
        # ollama_llm = OllamaService(
        #     base_url=OLLAMA_BASE_URL,
        #     username=OLLAMA_USERNAME,
        #     password=OLLAMA_PASSWORD,
        #     model=OLLAMA_MODEL
        # )
        from .chat_bot import get_model
        llm =get_model()  # Use the existing get_model function to maintain consistency and leverage any caching or configuration it provides
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        for tool_item in tools:
            log_info(
                f"[{tenant_id}] Tool {tool_item.name}: {tool_item.description}",
                tenant_id,
                conversation_id,
            )


        SQL_SYSTEM_PROMPT = """
            You are an agent designed to interact with a SQL database. Given an input question,
            create a syntactically correct {dialect} query, execute it, and return the answer.

            - Query only necessary columns.
            - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
            - **CRITICAL**: ONLY query columns that contain simple text or numerical data. Avoid querying columns that contain complex types like JSON, JSONB, or Arrays, as they cause internal errors.
            - Double-check your query before execution.
            - ALWAYS look at the tables first to understand the schema.
            - **CRITICAL SCHEMAS TO REFERENCE**: 
                - customer_account, customer_branchperformance, customer_contact, customer_conversation, 
                - customer_crmuser, customer_customer, customer_lead, customer_loanreport, 
                - customer_message, customer_opportunity, customer_transaction, org_location,
                - ats_jobposting, ats_application, org_jobrole, employees_employee, 
                - leave_leavetype, leave_leavebalance.
            - Use these tables to perform deep data analysis and visualizations.
            - Limit your query to at most 5 results.
            """
        # agent = create_agent(llm, tools, system_prompt=SQL_SYSTEM_PROMPT)
        # Use create_react_agent for better compatibility with LangGraph and stream/invoke methods
        agent = create_react_agent(
            llm,
            tools,
            prompt=SQL_SYSTEM_PROMPT.format(dialect=db.dialect),
        )
        log_info(
            f"[{tenant_id}] SQL Agent initialized successfully.",
            tenant_id,
            conversation_id,
        )
      
        
        TENANT_SQL_AGENTS[tenant_id] = agent
        TENANT_DBS[tenant_id] = db
        log_info(" SQL Agent initialized successfully.", tenant_id, conversation_id)
        return agent    

    except Exception as e:
        log_error(
            f"[{tenant_id}] Error initializing SQL Agent: {e}",
            tenant_id,
            conversation_id,
        )
        return None

@tool("get_customer_details_tool", args_schema=CustomerDetailsInput)
def get_customer_details_tool(config: RunnableConfig, **kwargs):
    """
    Retrieves a customer's details from the PostgreSQL database using phone number, email, or account number.
    """
    tenant_code = config["configurable"].get("tenant_id")
    tenant_id = tenant_code if tenant_code else "unknown"
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    val = kwargs.get('phone_or_email')
    acc_num = kwargs.get('account_number')
    log_info(f"get_customer_details_tool invoked with args: {val}, {acc_num}", tenant_id, conversation_id)
    tid = kwargs.get('current_tool_id') or "unknown_id"
    db_uri = config["configurable"].get("db_uri")
    
    
    if not val and not acc_num:
        return ToolMessage(content="Please provide an email, phone number, or account number.", tool_call_id=tid)

    try:
        if not db_uri:
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve Tenant ID
                t_query = text("SELECT id FROM org_tenant WHERE code = :code")
                tenant_id = connection.execute(t_query, {"code": tenant_code}).fetchone()
                tenant_id = tenant_id[0] if tenant_id else None

                sql = """
                    SELECT customer_id, first_name, last_name, email, phone_number, account_number, gender, occupation, nationality
                    FROM customer_customer
                    WHERE (email = :val OR phone_number = :val OR account_number = :acc)
                    AND tenant_id = :tid
                """
                
                res = connection.execute(text(sql), {"val": val, "acc": acc_num, "tid": tenant_id}).fetchone()
                
                if not res:
                    return ToolMessage(content="No customer found with the provided details.", tool_call_id=tid)
                
                details = (
                    f"Customer Found:\n"
                    f"- Name: {res.first_name} {res.last_name}\n"
                    f"- Email: {res.email}\n"
                    f"- Phone: {res.phone_number}\n"
                    f"- Account Number: **{res.account_number}**\n"
                    f"- Occupation: {res.occupation}\n"
                    f"- Nationality: {res.nationality}"
                )
                return ToolMessage(content=details, tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
        log_error(f"Failed to retrieve customer details: {str(e)}", tenant_id, conversation_id)
        return ToolMessage(content=f"Error retrieving customer details: {str(e)}", tool_call_id=tid)

@tool("update_customer_tool", args_schema=UpdateCustomerProfileInput)
def update_customer_tool(config: RunnableConfig, **kwargs):
    """
    Updates the customer's profile details (phone, email, occupation, etc.) in the PostgreSQL database.
    """
    tenant_code = config["configurable"].get("tenant_id")
    tenant_id = tenant_code if tenant_code else "unknown"
    conversation_id = config["configurable"].get("conversation_id", "unknown")
    target_email = kwargs.get('email') or config["configurable"].get("employee_id")
    target_phone = kwargs.get('phone_number')
    log_info(f"update_customer_tool invoked with args: {target_email}, {target_phone}", tenant_id, conversation_id)

    tid = kwargs.get('current_tool_id') or "unknown_id"
    db_uri = config["configurable"].get("db_uri")
    
    # We need a way to identify which customer to update. 
    # Since this is likely the current user, we assume their email/phone is in the context or provided.
    # For now, we'll use the provided email or phone if available, or fall back to the config employee_id (email).
    

    update_data = {k: v for k, v in kwargs.items() if v is not None and k != 'current_tool_id'}
    
    if not update_data:
        return ToolMessage(content="No update information was provided.", tool_call_id=tid)

    log_info(f"Updating customer profile for: {target_email or target_phone}", tenant_id, conversation_id)

    try:
        if not db_uri:
            log_error("Database configuration missing.", tenant_id, conversation_id)
            return ToolMessage(content="Error: Database configuration missing.", tool_call_id=tid)

        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)

        from sqlalchemy import text
        engine = create_engine(db_uri)
        
        try:
            with engine.connect() as connection:
                # 1. Resolve Tenant ID
                t_query = text("SELECT id FROM org_tenant WHERE code = :code")
                tenant_id = connection.execute(t_query, {"code": tenant_code}).fetchone()
                tenant_id = tenant_id[0] if tenant_id else None

                # 2. Build Dynamic Update SQL
                set_clauses = []
                params = {"tid": tenant_id, "target_email": target_email, "target_phone": target_phone}
                
                for key, val in update_data.items():
                    set_clauses.append(f"{key} = :{key}")
                    params[key] = val
                
                sql = f"""
                    UPDATE customer_customer 
                    SET {', '.join(set_clauses)}
                    WHERE (email = :target_email OR phone_number = :target_phone)
                    AND tenant_id = :tid
                """
                
                result = connection.execute(text(sql), params)
                connection.commit()
                
                if result.rowcount == 0:
                    return ToolMessage(content="No customer record found to update.", tool_call_id=tid)
                
                msg = f"Successfully updated your profile: {', '.join(update_data.keys())}."
                return ToolMessage(content=msg, tool_call_id=tid)
        finally:
            engine.dispose()

    except Exception as e:
       log_error(f"Failed to update customer profile: {str(e)}", tenant_id, conversation_id)
       return ToolMessage(content=f"Error updating customer profile: {str(e)}", tool_call_id=tid)




@tool(
    "sql_query_tool",
    description="Useful for answering questions requiring data from a SQL database (e.g., 'How many users are there?'). Input should be a natural language question.",
)
def sql_query_tool(query: str, state: dict) -> dict:
    # --- 1. State Extraction & Logging ---
    tenant_config = state.get("tenant_config", {})
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    db_uri = tenant_config.get("db_uri") or state.get("db_uri")

    log_info(f"sql_query_tool invoked for tenant {tenant_id}. Query: {query}", tenant_id, conversation_id)

    if not db_uri:
        log_warning("DB URI is missing. Tool unavailable.", tenant_id, conversation_id)
        return {"sql_result": "Db tool not available"}

    # --- 2. Isolated Initialization (Scorched Earth for Windows Stability) ---
    try:
        log_info("!!! TRACE: sql_query_tool - Entering isolated init block", tenant_id, conversation_id)
        # Create fresh DB and LLM instances for this specific tool call
        log_info("!!! TRACE: sql_query_tool - Importing SQLDatabase...", tenant_id, conversation_id)
        from langchain_community.utilities import SQLDatabase
        log_info("!!! TRACE: sql_query_tool - Importing SQLDatabaseToolkit...", tenant_id, conversation_id)
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        log_info("!!! TRACE: sql_query_tool - Importing create_react_agent...", tenant_id, conversation_id)
        from langgraph.prebuilt import create_react_agent
        
        if db_uri and db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)
            
        log_info(f"!!! TRACE: sql_query_tool - Calling SQLDatabase.from_uri with: {db_uri[:20]}...", tenant_id, conversation_id)
        db = SQLDatabase.from_uri(db_uri)
        log_info("!!! TRACE: sql_query_tool - SQLDatabase.from_uri SUCCESS", tenant_id, conversation_id)
        
        # Use a fresh OllamaService instance
        log_info("!!! TRACE: sql_query_tool - Initializing fresh OllamaService...", tenant_id, conversation_id)
        # ollama_llm = OllamaService(
        #     base_url=OLLAMA_BASE_URL,
        #     username=OLLAMA_USERNAME,
        #     password=OLLAMA_PASSWORD,
        #     model=OLLAMA_MODEL
        # )
        llm = get_model()
        log_info("!!! TRACE: sql_query_tool - OllamaService init SUCCESS", tenant_id, conversation_id)
        
        log_info("!!! TRACE: sql_query_tool - Initializing SQLDatabaseToolkit...", tenant_id, conversation_id)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_tools = toolkit.get_tools()
        log_info(f"!!! TRACE: sql_query_tool - Toolkit tools retrieved: {len(sql_tools)}", tenant_id, conversation_id)
        
        
        SQL_SYSTEM_PROMPT = """
            You are an agent designed to interact with a SQL database. Given an input question,
            create a syntactically correct {dialect} query, execute it, and return the answer.

            - Query only necessary columns.
            - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP).
            - **CRITICAL**: ONLY query columns that contain simple text or numerical data. Avoid querying columns that contain complex types like JSON, JSONB, or Arrays, as they cause internal errors.
            - Double-check your query before execution.
            - ALWAYS look at the tables first to understand the schema.
            - **CRITICAL SCHEMAS TO REFERENCE**: 
                - customer_account, customer_branchperformance, customer_contact, customer_conversation, 
                - customer_crmuser, customer_customer, customer_lead, customer_loanreport, 
                - customer_message, customer_opportunity, customer_transaction, org_location,
                - ats_jobposting, ats_application, org_jobrole, employees_employee, 
                - leave_leavetype, leave_leavebalance.
            - Use these tables to perform deep data analysis and visualizations.
            - Maximum number of turns of call is 3, otherwise return the result.
            - Limit your query to at most 5 results.
            - Return your final answer as a JSON object with keys: "analysis" (text summary) and "data" (list of dictionaries representing the rows).
            """
        
        # Initialize a fresh agent for this specific call
        log_info("!!! TRACE: sql_query_tool - Creating react agent...", tenant_id, conversation_id)
        agent = create_react_agent(
            llm,
            sql_tools,
            prompt=SQL_SYSTEM_PROMPT.format(dialect=db.dialect),
        )
        log_info("!!! TRACE: sql_query_tool - Agent creation SUCCESS", tenant_id, conversation_id)
        
        log_info(f"!!! CRASH TRACE: About to call agent.invoke() for query: {query}", tenant_id, conversation_id)
        agent_response = agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        log_info(f"!!! CRASH TRACE: agent_response: {agent_response}", tenant_id, conversation_id)
        log_info("!!! CRASH TRACE: agent.invoke() COMPLETED SUCCESSFULLY", tenant_id, conversation_id)
        
        agent_msgs = agent_response.get("messages", [])
        result = agent_msgs[-1].content if agent_msgs else "No response from SQL agent."
        
        log_info(f"DEBUG: SQL Agent response received. Length: {len(result)}", tenant_id, conversation_id)
        
        # Attempt to parse JSON containing 'analysis' and 'data'
        import json
        import re
        parsed_data = None
        analysis_text = result
        
        # Check if the result specifically looks like it contains encoded JSON
        try:
            # Look for JSON block if wrapped in markdown
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result, re.DOTALL)
            json_str = match.group(1) if match else result
            
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                if 'data' in parsed:
                    parsed_data = parsed['data']
                    if not isinstance(parsed_data, list):
                        parsed_data = [parsed_data]  # Ensure list
                analysis_text = parsed.get('analysis', "Here is the data you requested.")
        except Exception:
            # If parsing fails, it's likely just a text/markdown response. We use it as is.
            pass
        log_info(f"DEBUG: SQL Agent response received. Length: {len(analysis_text)}", tenant_id, conversation_id)   
        log_info(f"DEBUG: SQL Agent response received. analysis: {analysis_text} and data: {parsed_data}", tenant_id, conversation_id) 
        return {"sql_result": analysis_text, "data": parsed_data}

    except BaseException as e:
        import traceback
        tb_str = traceback.format_exc()
        log_error(f"CRITICAL ERROR in sql_query_tool:\nType: {type(e)}\nError: {e}\nTraceback: {tb_str}", tenant_id, conversation_id)
        return {"sql_result": f"Error executing SQL query: {e}. Check server logs."}

@tool(
    "pdf_retrieval_tool",
    description="Useful for answering questions from the orgabisation's internal knowledge base (PDFs). Input should be a specific question.",
    #  args_schema=ToolInput # Apply the schema
)
def pdf_retrieval_tool(query: str, state: dict) -> dict:
    """
    Performs a document query using the pre-initialized FAISS vector store.

    Returns:
        dict: A dictionary containing 'pdf_content'. This will hold the search
              results on success, or a structured error message on failure.
    """
    tenant_id = state.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    tenant_id = state.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    log_info(f"pdf_retrieval_tool invoked with query: {query}", tenant_id, conversation_id)
    if "state" in state:
        state = state["state"]

    tenant_id = "unknown"
    conversation_id = "unknown"

    if isinstance(state, str):
        try:
            state = json.loads(state)
            log_info(
                "State JSON decoded for pdf_retrieval_tool successfully.",
                "unknown",
                "unknown",
            )

        except json.JSONDecodeError as e:
            log_error(
                f"Invalid input format. JSON decode failed: {e}",
                tenant_id,
                conversation_id,
            )
            return {
                "pdf_content": "!!ERROR!! CODE:PDF-4001 MESSAGE:Invalid input format (expected dictionary/JSON) for PDF search."
            }
    user_query = query.strip()
    # Extract Contextual IDs for Logging
    tenant_config = state.get("tenant_config", {})
    # user_query = state.get("user_query", "").strip()
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    vector_store_path = state.get("vector_store_path")
    log_info(f"retrieve_from_pdf tool invoked with query {query}", tenant_id, conversation_id)
    # log_info("search_pdf tool invoked", tenant_id, conversation_id)

    # 2. Handle missing query
    if not user_query:
        log_warning(
            "No user query provided for PDF search.", tenant_id, conversation_id
        )
        return {
            "pdf_content": "!!ERROR!! CODE:PDF-4002 MESSAGE:No user query provided for PDF search."
        }

    # 3. Handle missing vector store path
    if not vector_store_path:
        log_error(
            "Missing vector_store_path in state. Cannot search.",
            tenant_id,
            conversation_id,
        )
        return {
            "pdf_content": "!!ERROR!! CODE:PDF-4003 MESSAGE:Vector store path not provided in state. Index is unreachable."
        }

    log_debug(f"Search Path: {vector_store_path}", tenant_id, conversation_id)

    try:
        # 4. LOAD FAISS Index
        log_info("Attempting to load FAISS index.", tenant_id, conversation_id)
        vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # 5. Check Index Count
        search_count = vector_store.index.ntotal
        log_info(
            f"FAISS index loaded successfully. Vector count: {search_count}",
            tenant_id,
            conversation_id,
        )

        if search_count == 0:
            log_warning(
                "Vector store is empty (0 documents).", tenant_id, conversation_id
            )
            return {
                "pdf_content": "!!ERROR!! CODE:PDF-4004 MESSAGE:Vector store loaded successfully but is empty (0 documents)."
            }

        # 6. Perform Similarity Search
        log_info(
            f"Performing similarity search for query: '{user_query[:50]}...'",
            tenant_id,
            conversation_id,
        )
        docs = vector_store.similarity_search(user_query, k=3)
        results_text = []
        sources = set()
        # content = "\n\n".join([doc.page_content for doc in results])
        log_info(
            f"Raw Output of Similarity Search : '{user_query[:50]}...'",
            tenant_id,
            conversation_id,
        )
        log_info(
            f"Full Raw Output of Similarity Search : '{user_query}...'",
            tenant_id,
            conversation_id,
        )
        for doc in docs:
            results_text.append(doc.page_content)
            source_file = os.path.basename(
                doc.metadata.get("source", "General Document")
            )
            sources.add(source_file)

        formatted_result = "\n\n".join(results_text)

        log_debug(
            f"PDF search returned {len(formatted_result)} results.",
            tenant_id,
            conversation_id,
        )
        # return {"pdf_content": formatted_content}
        # return {"type": "pdf_content", "content": formatted_content}
        # return {formatted_content}
        # We return a structured string that the LLM can easily see sources from
        return {"pdf_content": formatted_result, "source_documents": list(sources)}

    except Exception as e:
        log_error(f"PDF Retrieval Error: {str(e)}", tenant_id, conversation_id)
        return {"pdf_content": f"!!ERROR!! CODE:PDF-5001 MESSAGE: {str(e)}"}


@tool("web_search_tool")
def web_search_tool(query: str, state: dict) -> dict:
    """
    Performs web search using Tavily.

    Returns:
        dict: A dictionary containing 'web_content'. This will hold the search
              results on success, or a structured error message on failure.
    """

    log_info("retrieve_from_web tool invoked", "unknown", "unknown")
    if "state" in state:
        state = state["state"]
    user_query = state.get("user_query", "")
    tenant_config = state.get("tenant_config", {})
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")

    log_info("search_web tool invoked", tenant_id, conversation_id)

    # 1. Collect priority domains from config
    priority_domains = []
    for field in ["tenant_knowledge_base", "tenant_website"]:
        url = tenant_config.get(field)
        if url:
            try:
                domain = urlparse(url).netloc
                if domain and domain not in priority_domains:
                    priority_domains.append(domain)
            except Exception:
                continue

    # 2. Step 1: Targeted Search (Restricted to priority domains)
    search_results = []
    source_label = "GENERAL SEARCH"  # Default label
    if priority_domains:
        log_info(
            f"Step 1: Searching priority domains {priority_domains}",
            tenant_id,
            conversation_id,
        )
        kb_tool = TavilySearch(
            max_results=5, include_domains=priority_domains, search_depth="advanced"
        )

        try:
            search_results = kb_tool.invoke({"query": user_query})
            if search_results:
                source_label = "TENANT WEBSITE/KNOWLEDGE_BASE"  # Tag as authoritative
        except Exception as e:
            log_warning(f"Priority search failed: {e}", tenant_id, conversation_id)

    # 3. Step 2: Fallback Logic (General search if zero results found)
    if not search_results:
        log_info(
            "Step 2: No results found in KB/Website. Falling back to general web search.",
            tenant_id,
            conversation_id,
        )
        general_tool = TavilySearch(max_results=5, search_depth="advanced")
        try:
            search_results = general_tool.invoke({"query": user_query})
            source_label = "GENERAL"  # Tag as external
        except Exception as e:
            log_error(f"Fallback search failed: {e}", tenant_id, conversation_id)

    # 4. Format Output
    # Extract results from Tavily response (handle both dict and error cases)
    if isinstance(search_results, dict) and "results" in search_results:
        results_list = search_results["results"]
    elif isinstance(search_results, list):
        results_list = search_results
    else:
        results_list = []
    
    # Format Output with Source Weighting Metadata
    formatted_parts = []
    for res in results_list:
        if isinstance(res, dict) and 'url' in res and 'content' in res:
            part = (
                f"--- SOURCE TYPE: {source_label} ---\n"
                f"URL: {res['url']}\n"
                f"CONTENT: {res['content']}\n"
            )
            formatted_parts.append(part)
    
    return {
        "web_content": "\n".join(formatted_parts) if formatted_parts else "!!ERROR!! CODE:WEB-4002 MESSAGE:No valid search results found.",
        "type": "web_search",
        "source_documents": source_label.lower(),
        "source_labels": [f"web_search_{source_label.lower()}"],
    }



# sql_query_tool = Tool(
#     name="sql_query_tool",
#     description="Useful for answering questions requiring data from a SQL database (e.g., 'How many users are there?'). Input should be a natural language question.",
#     func=execute_sql_query_func,
#     args_schema=SQLQueryInput,
# )


@tool("generate_visualization_tool", args_schema=VisualizationInput, description="Use this tool to create charts, graphs, plots, or any data visualizations. This is the best tool when the user asks to 'plot', 'chart', 'visualize', or 'draw' data.")
def generate_visualization_tool(query: str, data: Any = None, state: Optional[dict] = None, **kwargs) -> dict:
    """
    Generates a data visualization based on a natural language query.
    """
    if state is None:
        state = {}
    tenant_config = state.get("tenant_config", {})
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    user_query = state.get("user_query", query)
    tid = kwargs.get('current_tool_id') or "unknown_id"
    log_info(f"Generating Visualization for query: '{query}'", tenant_id, conversation_id)
    analysis_text = "" # Initialize in case of early failure
    try:
        from .chat_bot import get_llm_instance
        llm = get_llm_instance()

        if data:
            log_info("Data provided directly to visualization tool. Bypassing SQL generation.", tenant_id, conversation_id)
            import json
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            df = pd.DataFrame(data)
            sql_query = "Data provided directly via tool chaining"
            if df.empty:
                log_warning("Data provided is empty.", tenant_id, conversation_id)
                return {"visualization_result": {"analysis": "No data available to visualize.", "image_base64": None}}
        else:
            # Step 2: Execute the query with Pandas
            tenant_config = state.get("tenant_config", {})
            tenant_id = tenant_config.get("tenant_id", "unknown")
            db_uri = tenant_config.get("db_uri") or DB_URI
            log_info(f"Database URI: '{db_uri}'", tenant_id, conversation_id)
            
            db = TENANT_DBS.get(tenant_id)
            if not db:
                from langchain_community.utilities import SQLDatabase
                db = SQLDatabase.from_uri(db_uri)
                TENANT_DBS[tenant_id] = db

            # Step 1: Generate SQL from the natural language query (with few-shot prompt)
            sql_generation_prompt = f"""Given the user's question {user_query}, create a single, syntactically correct SQL query to retrieve the data needed for a chart.
                    Do not include any other text or explanation, just the SQL query itself.

                    Tables available: {db.get_table_info()}

                    ### Example ###
                    User question: "Show me the total transaction value for each month this year."
                    SQL Query:
                    ```sql
                    SELECT
                    STRFTIME('%Y-%m', timestamp) AS month,
                    SUM(amount) AS total_value
                    FROM
                    ai_transaction
                    WHERE
                    STRFTIME('%Y', timestamp) = STRFTIME('%Y', 'now')
                    GROUP BY
                    month
                    ORDER BY
                    month;
                    ```
                    ### End Example ###

                    User question: "{query}"
                    SQL Query:
                    """
            raw_sql_query = llm.invoke(sql_generation_prompt).content.strip()
            
            match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_sql_query, re.DOTALL)
            if match:
                sql_query = match.group(1).strip()
            else:
                sql_query = raw_sql_query
            
            log_info(f"Generated SQL: {sql_query}", tenant_id, conversation_id)

            # Step 2: Execute the query with Pandas
            engine = create_engine(db_uri)
            df = pd.read_sql_query(sql_query, con=engine)
            
        if df.empty:
            log_warning("Query returned no data.", tenant_id, conversation_id)
            return {"visualization_result": {"analysis": "I found no data to visualize for your request.", "image_base64": None}}

        # --- ENHANCED LOGGING: Log DataFrame details ---
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        log_info(f"--- DataFrame Details ---\nHead:\n{df.head().to_string()}\nInfo:\n{df_info_str}", tenant_id, conversation_id)

        # Step 3: Determine the best chart type
        df_info_for_prompt = f"Data Columns: {df.columns.tolist()}\nData Head:\n{df.head().to_string()}"
        chart_selection_prompt = f"""
        Given the user's original query '{query}' and the following data summary, what is the best chart type to use?
        Your answer must be a single word from this list: 'bar', 'line', 'scatter', 'pie'.

        Data Summary:\n{df_info_for_prompt}
        """
        chart_type = llm.invoke(chart_selection_prompt).content.strip().lower()
        log_info(f"LLM chose chart type: '{chart_type}'", tenant_id, conversation_id)

        # Step 4: Get textual analysis from the LLM
        analysis_prompt = f"Analyze this data and provide a brief, insightful summary based on the user's original request: '{query}'.\n\nData:\n{df.to_csv(index=False)}"        
        analysis_text = llm.invoke(analysis_prompt).content
        log_info(f"Generated Analysis: {analysis_text[:200]}...", tenant_id, conversation_id) # Log a snippet

        # Step 5: Generate the plot using intelligent chart selection
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric, categorical, dates = get_column_types(df)
        
        if chart_type == 'bar' and categorical and numeric:
            x_col = categorical[0]
            if len(numeric) > 1: # Handle multi-series bar charts
                df.set_index(x_col)[numeric].plot(kind='bar', ax=ax, figsize=(12, 7))
                ax.set_ylabel("Values")
                ax.legend(title='Metrics')
            else: # Handle single-series bar charts
                y_col = numeric[0]
                df.plot(kind='bar', x=x_col, y=y_col, ax=ax, legend=False)
                ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_xlabel(x_col.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'line' and (dates or numeric):
            x_col = dates[0] if dates else numeric[0]
            y_cols = [c for c in numeric if c != x_col]
            if not y_cols: y_cols = numeric # Fallback if x is also the only numeric
            df.plot(kind='line', x=x_col, y=y_cols, ax=ax, marker='o')
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'scatter' and len(numeric) >= 2:
            x_col, y_col = numeric[0], numeric[1]
            df.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())

        elif chart_type == 'pie' and categorical and numeric:
            df.set_index(categorical[0])[numeric[0]].plot(
                kind='pie', ax=ax, autopct='%1.1f%%', startangle=90
            )
            ax.set_ylabel('')
        
        else: # Fallback
            log_warning(f"Could not find a perfect chart match for type '{chart_type}'. Using generic plot.", tenant_id, conversation_id)
            df.plot(ax=ax)
       
        # Formatting common to all charts
        ax.set_title(query.title())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        
        # Step 6: Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        log_info(f"Successfully generated plot image (Base64 length: {len(image_base64)}).", tenant_id, conversation_id)
    except Exception as e:
        # --- ENHANCED LOGGING: Log the full exception traceback ---
        log_error(f"Error in visualization tool: {e}", tenant_id, conversation_id)
        analysis_text_on_error = analysis_text if analysis_text else f"Sorry, I encountered an unrecoverable error: {e}"
        return {"visualization_result": {"analysis": analysis_text_on_error, "image_base64": None}}
     
    # return {"visualization_result": {"analysis": analysis_text, "image_base64": image_base64}}   
    log_info(f"Successfully generated plot image. Base64 size: {len(image_base64)} bytes.", tenant_id, conversation_id)

    return Command(
        update={
            "visualization_image": image_base64,
            "visualization_analysis": analysis_text,
            "messages": [
                ToolMessage(
                    content=f"Visualization generated successfully. Analysis: {analysis_text}",
                    tool_call_id=tid,
                    name="generate_visualization_tool"
                )
            ]
        }
    )

        
       
   


# --- FULLY ENHANCED VISUALIZATION TOOL ---



def get_column_types(df: pd.DataFrame):
    """Helper function to identify column types for plotting."""
    # Note: get_column_types is a helper and doesn't have easy access to tenant_id/conversation_id 
    # without passing them through. We'll leave this one as a simple debug or remove it.
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols

tools = [
    get_payslip_tool,
    fetch_available_leave_types_tool,
    validate_leave_balance_tool,
    prepare_leave_application_tool,
    calculate_num_of_days_tool,
    submit_leave_application_tool,
    search_job_opportunities_tool,
    fetch_leave_status_tool,
    search_travel_deals_tool,
    create_customer_profile_tool,
    get_customer_details_tool,
    generate_visualization_tool,
    sql_query_tool,
    pdf_retrieval_tool,
    web_search_tool,
]