import os
import sys
import json
import uuid
import base64
import logging
import re
import io
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Union, Annotated, cast
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from logging.handlers import RotatingFileHandler

# LangChain / LangGraph
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredFileLoader
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

# Local Imports
from .database import SessionLocal
from .ollama_service import OllamaService
from .base import State, Answer
from .tools import tools, init_sql_agent, TENANT_SQL_AGENTS

load_dotenv()

# Logging setup
logger = logging.getLogger("HR_AGENT")

def log_info(msg, tenant_id, conversation_id):
    logger.info(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

def log_error(msg, tenant_id, conversation_id):
    logger.error(f"[Tenant: {tenant_id} | Conversation: {conversation_id}] {msg}")

# Constants / Fallbacks
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
# DEFAULT_AGENT_PROMPT = 


GLOBAL_ROUTING_PROMPT = """You are a helpful AI assistant for ATB Bank. Your task is to analyze the user's request and decide if a tool is needed to answer it.

You have access to the following tools:
- `pdf_retrieval_tool`: For questions about bank policies, products, or internal knowledge.
- `tavily_search_tool`: For general knowledge or up-to-date information.
- `sql_query_tool`: For questions about specific data, like user counts or transaction volumes.
- **`generate_visualization_tool`**: **Use this tool ONLY when the user explicitly asks to 'plot', 'chart', 'graph', or 'visualize' data. This is your primary tool for creating visual representations from database data.**

Based on the conversation history, either call the most appropriate tool to gather information or, if you have enough information already, prepare to answer the user directly.
"""

GLOBAL_FINAL_ANSWER_PROMPT1 = """You are Damilola, the AI-powered virtual assistant for ATB. Your role is to deliver professional customer service and insightful data analysis, depending on the user's needs.

You operate in three modes:
1. **Customer Support**: Respond with empathy, clarity, and professionalism. Your goal is to resolve issues, answer questions, and guide users to helpful resources — without technical jargon or internal system references.
2. **Data Analyst**: Interpret data, explain trends, and offer actionable insights. When visualizations are included, describe what the chart shows and what it means for the user.
3. **HR Assistant**: Respond with empathy, clarity, and professionalism regarding leave, payslips, and workplace policies.

Your response must be:
- **Final**: No follow-up questions or uncertainty.
- **Clear and Polite**: Use emotionally intelligent language, especially if the user expresses frustration or confusion.
- **Context-Aware**: Avoid mentioning internal systems (e.g., database names or SQL sources) .
- **Structured**: Always return your answer in the following JSON format.
- **Structured**: use naira sign whne currency us required 

"""

GLOBAL_FINAL_ANSWER_PROMPT = """
  You are Damilola, the AI-powered virtual assistant for ATB. Your role is to deliver professional customer service and insightful data analysis, depending on the user's needs.

You operate in three modes:
1. **Customer Support**: Respond with empathy, clarity, and professionalism. Your goal is to resolve issues, answer questions, and guide users to helpful resources — without technical jargon or internal system references.
2. **Data Analyst**: Interpret data, explain trends, and offer actionable insights. When visualizations are included, describe what the chart shows and what it means for the user.
3. **HR Assistant**: Respond with empathy, clarity, and professionalism regarding leave, payslips, and workplace policies.

Your response must be:
- **Final**: No follow-up questions or uncertainty.
- **Clear and Polite**: Use emotionally intelligent language, especially if the user expresses frustration or confusion.
- **Context-Aware**: Avoid mentioning internal systems (e.g., database names or SQL sources) .
- **Structured**: Always return your answer in the following JSON format.
- **Structured**: use naira sign whne currency us required 

    OPERATING PROTOCOLS:
    
    PROTOCOL 1: LEAVE REQUESTS
    - If the user wants to apply for leave, you MUST first call 'fetch_available_leave_types_tool'.
    - If the user specifies a leave type NOT in the list provided by 'fetch_available_leave_types_tool':
      1. Politely inform them that '[InvalidType]' is not available for their category.
      2. Re-list the valid options.
      3. Do NOT call 'prepare_leave_application_tool' until a valid type is selected.
    - LEAVE YEAR LOGIC: Ask the user: "Is this leave for the current year or your previous year's carry-over?"
      Current -> {current_year}, Previous -> {previous_year}.
    - SUCCESS: After 'submit_leave_application_tool' confirms success, if it was 'Vacation', offer help with travel via 'search_travel_deals_tool'.

    PROTOCOL 2: PAYSLIPS
    - Once 'get_payslip_tool' returns, inform the user: 'Your payslip has been sent to your email.'

    PROTOCOL 3: HR POLICIES & KNOWLEDGE
    - For policy questions, use 'pdf_retrieval_tool' to search HR handbooks.

    PROTOCOL 4: DATA ANALYTICS
    - Use 'sql_query_tool' for data inquiries. Provide actionable insights.
    - Use 'generate_visualization_tool'  when user  asked to 'plot', 'chart', 'graph', or 'visualize'.

    PROTOCOL 5: PROFILE UPDATES
    - Use 'update_customer_tool' or 'update_employee_profile_tool'.
    - If bank name is missing for an account update, ask for it before calling the tool.

    PROTOCOL 6: LEAVE STATUS
    - Use 'fetch_leave_status_tool' for approvals and pending status.

    CONTEXT:
    - Employee ID: {ID}
    - Current Date: {current_date_str}
    - Current Leave/Workflow Status: {status_summary}
    - Document Context: {pdf_content}
    - Web Context: {web_content}
    - SQL Result: {sql_result}
    
    ### Output Format:
You MUST return ONLY a valid JSON object. Do not include any text outside the JSON block.
```json
{{
  "answer": "Your response to the user",
}}
```
    """



# DEFAULT_TOOL_INTENT_MAP = {
#     "leave_management": {"tools": ["fetch_available_leave_types_tool", "prepare_leave_application_tool"], "triggers": ["leave", "vacation"]},
#     "data_analysis": {"tools": ["sql_query_tool"], "triggers": ["report", "count", "average"]},
#     "visualization": {"tools": ["generate_visualization_tool"], "triggers": ["plot", "chart", "graph"]}
# }

TOOL_INTENT_MAP = {
    "leave_management": {
        "tools": ["fetch_available_leave_types_tool", "prepare_leave_application_tool", 
                  "submit_leave_application_tool", "fetch_leave_status_tool", "calculate_num_of_days_tool"],
        "triggers": ["leave", "vacation", "sick leave", "day off", "approve", "leave balance", "resumption"]
    },
    "payslip_services": {
        "tools": ["get_payslip_tool"],
        "triggers": ["payslip", "salary", "pay statement", "earnings", "payroll"]
    },
    "hr_policy": {
        "tools": ["pdf_retrieval_tool"],
        "triggers": ["policy", "handbook", "benefits", "hr guide", "rules"]
    },
    "data_analysis": {
        "tools": ["sql_query_tool"],
        "triggers": ["report", "count", "average", "total", "statistics", "data"]
    },
    "visualization": {
        "tools": ["generate_visualization_tool"],
        "triggers": ["plot", "chart", "graph", "visualize", "show as a bar chart"]
    },
    "profile_updates": {
        "tools": ["update_employee_profile_tool", "create_customer_profile_tool", "get_customer_details_tool"],
        "triggers": ["update", "profile", "phone number", "bank account", "details"]
    },
    "recruitment": {
        "tools": ["search_job_opportunities_tool"],
        "triggers": ["job", "vacancy", "career", "hiring", "position"]
    },
    "travel_concierge": {
        "tools": ["search_travel_deals_tool"],
        "triggers": ["flight", "hotel", "travel", "booking", "trip"]
    },
    "general_inquiry": {
        "tools": ["web_search_tool"],
        "triggers": ["what is", "how to", "who is", "search"]
    }
}


# DEFAULT_FINAL_ANSWER_PROMPT = """You are Damilola, the AI-powered virtual assistant. Deliver professional customer service and insightful data analysis."""

DEFAULT_EMPLOYEE_ID = "obinna.kelechi.adewale@dignityconcept.tech"

embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    return embeddings

# ---------------------------------------------------------------------------
# FAS Vector Store service URLs
# The Django service at this URL hosts the tenant's document corpus.
# Set VECTOR_STORE_LOCAL=true in .env to use localhost instead (for local dev).
# Override URLs via VECTOR_STORE_URL / VECTOR_STORE_LOCAL_URL env vars.
# ---------------------------------------------------------------------------
_FAS_REMOTE_URL = os.getenv(
    "VECTOR_STORE_URL",
    "http://147.182.194.8:3000/projects/whatsapp-1/app/vectra_app"
)
_FAS_LOCAL_URL = os.getenv(
    "VECTOR_STORE_LOCAL_URL",
    "http://localhost:3000/projects/whatsapp-1/app/vectra_app"
)
_USE_LOCAL_VECTOR = os.getenv("VECTOR_STORE_LOCAL", "false").lower() == "true"
_VECTOR_STORE_TIMEOUT = int(os.getenv("VECTOR_STORE_TIMEOUT", "10"))


def _fetch_docs_from_fas(tenant_id: str) -> list:
    """
    Try to fetch documents from the FAS Django service.
    Expects JSON: {"documents": [{"page_content": "...", "metadata": {...}}, ...]}

    Tries the remote URL first (unless VECTOR_STORE_LOCAL=true), then localhost.
    Returns a list of LangChain Document objects, or [] on any failure.
    """
    from langchain_core.documents import Document as LCDoc

    urls = []
    if not _USE_LOCAL_VECTOR:
        urls.append((_FAS_REMOTE_URL, "REMOTE"))
    urls.append((_FAS_LOCAL_URL, "LOCAL"))

    for url, label in urls:
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Trying {label} FAS at {url}")
        try:
            resp = requests.get(url, params={"tenant_id": tenant_id}, timeout=_VECTOR_STORE_TIMEOUT)
            resp.raise_for_status()
            raw_docs = resp.json().get("documents", [])

            if not isinstance(raw_docs, list):
                logger.warning(f"[VectorStore | Tenant: {tenant_id}] {label}: 'documents' is not a list.")
                continue

            docs = [
                LCDoc(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in raw_docs
                if isinstance(d, dict) and d.get("page_content")
            ]
            logger.info(f"[VectorStore | Tenant: {tenant_id}] {label}: fetched {len(docs)} docs.")
            return docs

        except requests.exceptions.ConnectionError:
            logger.warning(f"[VectorStore | Tenant: {tenant_id}] {label}: connection refused at {url}.")
        except requests.exceptions.Timeout:
            logger.warning(f"[VectorStore | Tenant: {tenant_id}] {label}: timed out after {_VECTOR_STORE_TIMEOUT}s.")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            logger.warning(f"[VectorStore | Tenant: {tenant_id}] {label}: HTTP {status} — {e}.")
        except Exception as e:
            logger.error(f"[VectorStore | Tenant: {tenant_id}] {label}: unexpected error — {e}.")

    logger.warning(f"[VectorStore | Tenant: {tenant_id}] All FAS endpoints failed.")
    return []


def initialize_vector_store(tenant_id: str):
    """
    Fetch and return a FAISS vector store for the given tenant.
    FastAPI is read-only — it never creates or saves a vector store.

    Resolution order:
      1. Fetch documents from remote Django FAS service → build in-memory FAISS.
      2. Fetch documents from localhost FAS service (fallback / local dev).
      3. Load a previously cached FAISS index from disk (faiss_dbs/<tenant_id>/).
      4. Return (None, info) if everything fails — no empty store is created.
    """
    tenant_id = str(tenant_id)
    persist_directory = os.path.join("faiss_dbs", tenant_id)

    logger.info(f"[VectorStore | Tenant: {tenant_id}] Initializing (read-only).")

    # --- Step 1: Embeddings ---
    try:
        emb = get_embeddings()
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Embeddings ready.")
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Embeddings init failed: {e}")
        return None, {"status": "error", "message": f"Embeddings init failed: {e}"}

    # --- Step 2: Try to fetch docs from FAS service (remote → local) ---
    try:
        docs = _fetch_docs_from_fas(tenant_id)
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] _fetch_docs_from_fas raised: {e}")
        docs = []

    if docs:
        try:
            vector_store = FAISS.from_documents(docs, emb)
            logger.info(f"[VectorStore | Tenant: {tenant_id}] In-memory FAISS built from {len(docs)} FAS docs.")
            return vector_store, {"status": "success", "source": "fas_service", "doc_count": len(docs)}
        except Exception as e:
            logger.error(f"[VectorStore | Tenant: {tenant_id}] Failed to build FAISS from FAS docs: {e}")

    # --- Step 3: Fall back to disk cache (built during a previous session or by Django) ---
    try:
        vector_store = FAISS.load_local(persist_directory, emb, allow_dangerous_deserialization=True)
        doc_count = vector_store.index.ntotal if hasattr(vector_store, "index") else "unknown"
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Loaded from disk cache. index_size={doc_count}")
        return vector_store, {"status": "success", "source": "disk_cache", "doc_count": doc_count}
    except FileNotFoundError:
        logger.warning(
            f"[VectorStore | Tenant: {tenant_id}] No disk cache at {persist_directory} and FAS unavailable. "
            "pdf_retrieval_tool will be disabled until the Django service is reachable."
        )
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Failed to load disk cache: {e}")

    return None, {"status": "not_found", "message": "FAS service unreachable and no disk cache found."}








def get_llm_instance(tenant_id=None):
    """Fetches LLM config using raw SQL."""
    with SessionLocal() as session:
        sql = "SELECT name, model FROM customer_llm LIMIT 1"
        res = session.execute(text(sql)).fetchone()
        
        if not res:
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        name = res[0].lower() if res[0] else "gemini"
        model_name = res[1] or "gemini-1.5-flash"
        
        if name == "gemini":
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
        elif "ollama" in name:
            return OllamaService(model=model_name)
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Graph Nodes
def extract_final_answer(response):
    """
    Robustly extract the final text answer from an LLM response or structured object.
    """
    # Case 1: Structured Answer or dict
    if isinstance(response, dict):
        return response.get("answer") or json.dumps(response)
    
    # Case 2: AIMessage or object with .content
    if hasattr(response, "content") and response.content:
        content = response.content
    else:
        # Fallback if content is empty or response is not a message object
        if hasattr(response, "tool_calls") and response.tool_calls:
            return "Executing tools..."
        content = str(response) if response is not None else "LLM did not return a response"

    # Clean up markdown and extract JSON blocks if present
    content_clean = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
    
    # Try direct parse
    try:
        parsed = json.loads(content_clean)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    except:
        pass

    # Try searching for JSON-like blocks if direct parse failed
    json_blocks = re.findall(r"\{.*?\}", content_clean, flags=re.DOTALL)
    for block in json_blocks:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "answer" in obj:
                return obj["answer"]
        except:
            continue

    return content_clean or "LLM returned empty response"

def assistant_node(state: State, config: RunnableConfig):
    # Fix for lint: "get" is not a known attribute of "None"
    if state is None:
        state = cast(State, {}) 
        
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    log_info(f"Assistant node triggered for tenant: {tenant_id}", tenant_id, conversation_id)

    tenant_config = state.get("tenant_config", {}) or {}
    push_name = tenant_config.get("push_name", "User")
    
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("thread_id", "unknown")
    log_info(f"Assistant node triggered for tenant: {tenant_id}", tenant_id, conversation_id)

    # --- Resolve DB-sourced prompts with hardcoded fallbacks ---
    tenant_config = state.get("tenant_config") or {}
    employee_id = {state.get('employee_id')}
    current_year = datetime.now().year
    previous_year = current_year - 1
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    status_summary = state.get("status_summary", "No active application.")
    pdf_content = state.get("pdf_content", "None")
    web_content = state.get("web_content", "None")
    sql_result = state.get("sql_result", "None")

    # global_answer_prompt acts as the persona/intro section of system_prompt.
    # Falls back to GLOBAL_FINAL_ANSWER_PROMPT if not set in the DB.
    global_answer_prompt = (
        tenant_config.get("global_answer_prompt")
        or GLOBAL_FINAL_ANSWER_PROMPT
    )
    agent_prompt = tenant_config.get("agent_prompt",GLOBAL_FINAL_ANSWER_PROMPT)

    if agent_prompt:
        system_prompt = agent_prompt.format(
            ID=employee_id,
            current_year=current_year,
            previous_year=previous_year,
            current_date_str=current_date_str,
            pdf_content=pdf_content,
            web_content=web_content,
            # Using leave_application for the state result as requested
            sql_result=sql_result,
            status_summary=status_summary
        )
    else:
        # Handle the case where no prompt is found
        system_prompt = "Default fallback prompt or error handling logic here."

       # 1. DYNAMIC CONTEXT PREPARATION (HR/Leave Status)
    leave_app = state.get("leave_application")
    status_summary = "No active application."
    if leave_app:
        status = leave_app.get("status")
        if status == "success":
            status_summary = f"Application {leave_app.get('application_id')} completed."
        elif status == "prepared":
            resumption = leave_app.get("details", {}).get("resumptionDate", "TBD")
            status_summary = f"Application prepared. Action required: Confirm resumption date {resumption}."
        elif status == "error":
            status_summary = f"Process error: {leave_app.get('message')}"



    # agent_prompt = tenant_config.get("agent_prompt") or DEFAULT_AGENT_PROMPT
    # final_answer_prompt = tenant_config.get("final_answer_prompt") or DEFAULT_FINAL_ANSWER_PROMPT
    
    # Static system prompt instruction for pushName
    greeting_instruction = f"Always greet or address the user using their name: {push_name}, if they are starting a conversation or if appropriate."
    
    llm = get_llm_instance(tenant_id)
    
    messages = state.get("messages", [])
    logger.info(f"Messages before assistant processing: {messages}")

    llm_with_tools = llm.bind_tools(tools)
    
  
    if messages and isinstance(messages[-1], ToolMessage):
        logger.info("Last message was a tool result. Preparing to generate final answer with structured output.")
        # Generate structured final answer
        logger.info("Generating final structured answer.")
        structured_llm = llm.with_structured_output(Answer)
        try:
            # final_answer_obj = structured_llm.invoke([SystemMessage(content=system_prompt)] + messages)
            unstructured_response = llm.invoke([SystemMessage(content=f"{system_prompt}\n\n{greeting_instruction}")] + messages)
            
            if hasattr(unstructured_response, "tool_calls") and unstructured_response.tool_calls:
                logger.info(f"Tool calls foundAtejjd: {unstructured_response.tool_calls}")
                return {"messages": [unstructured_response]}  # keep the AIMessage intact
            
        except Exception as e:
            log_error(f"Structured output generation failed: {e}", tenant_id, conversation_id)
            unstructured_response = None

        if unstructured_response:
            # Attach chart if present in state
            viz_result = state.get("visualization_result")
            if viz_result and "image_base64" in viz_result:
                unstructured_response.chart_base64 = viz_result["image_base64"]

            return {
                "messages": [AIMessage(content=unstructured_response.content)],
                # "leave_application": unstructured_response.dict(),
                # "metadata": {"sentiment": unstructured_response.sentiment, "sources": unstructured_response.source}
            }
        else:
            # Fallback to extract from raw invoke if structured fails
            logger.warning("Falling back to raw extraction in assistant_node.")
            # response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
            response = llm.invoke([SystemMessage(content=f"{system_prompt}\n\n{greeting_instruction}")] + messages)
            final_answer = extract_final_answer(response)
            return {"messages": [AIMessage(content=final_answer)]}

    llm_with_tools = llm.bind_tools(tools)
    logger.info("Invoking LLM with tools for assistant response generation.")
    # Inside assistant_node, right after the LLM call:

    
    system_msg = SystemMessage(content=f"{system_prompt}\n\n{greeting_instruction}")
    # system_msg = SystemMessage(content=f"{agent_prompt}\n\n{greeting_instruction}\n\n{final_answer_prompt}")
    response = llm_with_tools.invoke([system_msg] + messages)
    
    # if isinstance(response.content, str) and not response.tool_calls:
    #     json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
    #     if json_match:
    #         try:
    #             parsed = json.loads(json_match.group())
    #             t_name = parsed.get("name") or parsed.get("tool")
    #             if t_name:
    #                 response.tool_calls = [{"name": t_name, "args": parsed.get("args", {}), "id": str(uuid.uuid4()), "type": "tool_call"}]
    #         except: pass

    # return {"messages": [response]}
      # 2. Check for "Embedded" Tool Calls (The Ollama Fix)
    if isinstance(response.content, str) and not response.tool_calls:
        try:
            # Search for JSON-like patterns in the text
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Map various LLM hallucinations to standard LangChain format
                t_name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
                t_args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters") or {}
                
                if t_name:
                    logger.info(f"Fixed: Extracted '{t_name}' from raw text content.")
                    # IMPORTANT: Manually populate the tool_calls attribute
                    response.tool_calls = [{
                        "name": t_name,
                        "args": t_args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    }]
        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")

    # 3. Now LangGraph will see response.tool_calls and route to tool_node
    if response.tool_calls:
        return {"messages": [response]}

    
    # 2. Check for "Embedded" Tool Calls (The Ollama Fix)
    if isinstance(response.content, str) and not response.tool_calls:
        try:
            # Search for JSON-like patterns in the text
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Map various LLM hallucinations to standard LangChain format
                t_name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
                t_args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters") or {}
                
                if t_name:
                    logger.info(f"Fixed: Extracted '{t_name}' from raw text content.")
                    # IMPORTANT: Manually populate the tool_calls attribute
                    response.tool_calls = [{
                        "name": t_name,
                        "args": t_args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    }]
        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")
        
    #     # 3. Now LangGraph will see response.tool_calls and route to tool_node
        if response.tool_calls:
            return {"messages": [response]}
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Tool calls foundAtejjdy: {response.tool_calls}")
            return {"messages": [response]}  # keep the AIMessage intact
        

    #     # Check 2: JSON-wrapped/Embedded Tool Calls (The "Ollama Fallback")
        if isinstance(response.content, str):
            try:
                # Look for JSON blocks even if mixed with text
                json_blocks = re.findall(r"\{.*?\}", response.content, flags=re.DOTALL)
                extracted_calls = []
                
                for block in json_blocks:
                    try:
                        parsed = json.loads(block)
                        if isinstance(parsed, dict):
                            # Support various formats:
                            # 1. Standard {'name': ..., 'args': ...}
                            # 2. Ollama Cloud {'tool': ..., 'parameters': ...}
                            # 3. List wrapper {'tool_calls': [...]}
                            
                            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                                extracted_calls.extend(parsed["tool_calls"])
                            else:
                                extracted_calls.append(parsed)
                    except:
                        continue
                
                if extracted_calls:
                    valid_calls = []
                    for call in extracted_calls:
                        # Detect tool name from various possible keys
                        t_name = call.get("name") or call.get("tool") or call.get("tool_name")
                        if not t_name: continue
                        
                        # Detect arguments from various possible keys
                        t_args = call.get("args") or call.get("parameters") or call.get("arguments") or {}
                        
                        valid_calls.append({
                            "name": t_name,
                            "args": t_args,
                            "id": call.get("id") or str(uuid.uuid4()),
                            "type": "tool_call"
                        })
                    
                    if valid_calls:
                        logger.info(f"Manually extracted {len(valid_calls)} tool calls from mixed content.")
                        response.tool_calls = valid_calls
                        # Also clean up the content to keep only the tool call if it's primarily a tool request
                        return {"messages": [response]}
            except Exception as e:
                logger.error(f"Failed to robustly parse tool calls from content: {e}")

    #     # --- END OF NEW PARSING ---

        final_answer = extract_final_answer(response)
        logger.info(f"LLM response Assitant Node: {final_answer}")
        return {"messages": [AIMessage(content=final_answer)]}




        # final_answer = extract_final_answer(response)
        # logger.info(f"LLM response Assitant Node: {final_answer}")
        # logger.info(f"Final Output Raw Aliko: {messages}")
        
        # return {"messages": [AIMessage(content=final_answer)]}


        # return {"messages": [final_answer]}


def tool_node(state: State):
    last_msg = state["messages"][-1]
    new_messages = []
    tools_by_name = {t.name: t for t in tools}
    
    for call in last_msg.tool_calls:
        tool_item = tools_by_name.get(call["name"])
        if tool_item:
            obs = tool_item.invoke({**call["args"], "state": state})
            new_messages.append(ToolMessage(content=str(obs), tool_call_id=call["id"]))
    
    return {"messages": new_messages}

def build_graph(checkpointer=None):
    workflow = StateGraph(State)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges(
        "assistant",
        lambda x: "tools" if x["messages"][-1].tool_calls else END
    )
    workflow.add_edge("tools", "assistant")
    
    return workflow.compile(checkpointer=checkpointer)

def process_message(message_content: str, conversation_id: str, tenant_id: str, employee_id: Optional[str] = None, push_name: str = "User"):
    # Fallback for employee_id
    if not employee_id:
        employee_id = DEFAULT_EMPLOYEE_ID
    log_info("Starting message processing pipeline", tenant_id, conversation_id)
    log_info(f"Employee ID: {employee_id}-message Content: {message_content}", tenant_id, conversation_id)
    
    vector_store_result = initialize_vector_store(tenant_id)

    # Unpack based on your initialize_vector_store return (vector_store, info_dict)
    tenant_vector_store, vs_info = (
        vector_store_result
        if isinstance(vector_store_result, tuple)
        else (vector_store_result, {})
    )

    if tenant_vector_store is not None:
        try:
            document_count = tenant_vector_store.index.ntotal
            log_info(f"Vector store document count: {document_count}", tenant_id, conversation_id)
        except AttributeError:
            log_error("Unexpected vector store structure.", tenant_id, conversation_id)
    else:
        log_error("Vector store is None.", tenant_id, conversation_id)
        
    with SessionLocal() as session:
        # 1. Fetch Tenant AI config
        ta_sql = """
            SELECT prompt_template_id, db_uri 
            FROM customer_tenant_ai 
            WHERE tenant_id = (SELECT id FROM org_tenant WHERE code = :code)
        """
        ta_res = session.execute(text(ta_sql), {"code": tenant_id}).fetchone()
        
        prompt_id = ta_res[0] if ta_res else None
        db_uri = ta_res[1] if ta_res else None
        
        # 2. Fetch Prompt
        if prompt_id:
            p_sql = "SELECT agent_prompt, \"GLOBAL_FINAL_ANSWER_PROMPT\", \"TOOL_INTENT_MAP\" FROM customer_prompt WHERE id = :pid"
            
            log_info(f"Agent Prompt Fetched from db: {p_sql}", tenant_id, conversation_id)
            p_res = session.execute(text(p_sql), {"pid": prompt_id}).fetchone()
        else:
            # Fallback to 'standard'
            p_sql = "SELECT agent_prompt, \"GLOBAL_FINAL_ANSWER_PROMPT\", \"TOOL_INTENT_MAP\" FROM customer_prompt WHERE name = 'standard' LIMIT 1"
            log_error(f"Fall back Agent Prompt : {p_sql}", tenant_id, conversation_id)
            p_res = session.execute(text(p_sql)).fetchone()

        tenant_config_dict = {
            "tenant_id": tenant_id,
            "employee_id": employee_id,
            "db_uri": db_uri,
            "push_name": push_name,
            "agent_prompt": p_res[0] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
            "final_answer_prompt": p_res[1] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
            "tool_intent_map": p_res[2] if p_res else TOOL_INTENT_MAP,
            # Vector store loaded from disk (built by Django service)
            "vector_store": tenant_vector_store,
        }

    with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as cp:
    # with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as cp:
        cp.setup()
        app = build_graph(cp)
        
        config = {"configurable": {"thread_id": conversation_id, "tenant_id": tenant_id, "employee_id": employee_id}}
        initial_state = {
            "messages": [HumanMessage(content=message_content)],
            "tenant_config": tenant_config_dict
        }
        
        final_state = app.invoke(initial_state, config=config)
        answer = final_state["messages"][-1].content
        
        return {"answer": answer}