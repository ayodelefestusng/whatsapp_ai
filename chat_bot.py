from ast import If
from calendar import c
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
from langchain.agents.structured_output import ToolStrategy
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
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy

from langchain.agents import create_agent
# Local Imports
from database import SessionLocal
from ollama_service import OllamaService
from base import State, Answer, Context, ResponseFormat
from logger_utils import log_info, log_error, log_debug, log_warning, logger
# from llm_handler import get_model
from tools import tools, init_sql_agent
OLLAMA_BASE_URL = "https://ai.notchhr.io/api/chat/local"
OLLAMA_USERNAME = "ai-user"
OLLAMA_PASSWORD = "x2GS7jEF@#2T"
OLLAMA_MODEL = "gpt-oss-safeguard:20b"
GEMINI_INIT= os.getenv("GEMINI_INIT", "google_genai:gemini-flash-latest")

# Constants / Fallbacks
# OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
# DEFAULT_AGENT_PROMPT = 
llm_fallback = init_chat_model(GEMINI_INIT)
model = llm_fallback  # Consistent naming for the primary LLM
_llm = None
def get_llm_instance(llm_config=None):
    """
    Returns an LLM instance based on the provided configuration or global DB setting.
    
    Supported LLM types:
    - gemini: Google Gemini API
    - ollama: Local Ollama instance
    - ollama_cloud: Ollama Cloud API (requires OLLAMA_API_KEY)
    """
    # If explicit config passed, use it. Otherwise fetch global if needed.
    # Note: 'llm_config' here is expected to be a Django ORM object or None.
    with SessionLocal() as session:
        sql = "SELECT name, model FROM customer_llm LIMIT 1"
        res = session.execute(text(sql)).fetchone()
        
        # If no config is found in the DB, default to a safe fallback
        if not res:
            logger.warning("No LLM config found in DB, defaulting to Gemini.")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        name = res[0].lower() if res[0] else "gemini"
        model_name = res[1] or "gemini-1.5-flash"
        
        if "gemini" in name:
            return ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv("GOOGLE_API_KEY"))
            
        elif "ollama" in name:
            # Initialize OllamaService without local network parameters
            return OllamaService(model=model_name)
            
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

def get_model():
    """
    Lazy-loads the model and binds tools only when needed.
    """
    global _llm
    
    if _llm is not None:
        return _llm

    try:
        base_llm = get_llm_instance()
        
        if base_llm is not None:
            # Replaced with standardized tool list
            from tools import tools
            # _llm = base_llm.bind_tools(tools)
            logger.info("✅ Model and tools initialized successfully.")
            return base_llm
        else:
            logger.error("❌ Failed to initialize base LLM.")
            return None
    except Exception as e:
        logger.error(f"❌ Error initializing model/tools: {e}")
        return None



# def get_llm_instance(llm_config=None):
#     """
#     Returns an LLM instance based on the provided configuration or global DB setting.
    
#     Supported LLM types:
#     - gemini: Google Gemini API
#     - ollama: Local Ollama instance
#     - ollama_cloud: Ollama Cloud API (requires OLLAMA_API_KEY)
#     """
#     # If explicit config passed, use it. Otherwise fetch global if needed.
#     # Note: 'llm_config' here is expected to be a Django ORM object or None.
#     with SessionLocal() as session:
#         sql = "SELECT name, model FROM customer_llm LIMIT 1"
#         res = session.execute(text(sql)).fetchone()
        
#         if not res:
#             return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        
#         name = res[0].lower() if res[0] else "gemini"
#         model_name = res[1] or "gemini-1.5-flash"
        
#     # logger.info("🌐 get_llm_instance ")
#     # if not llm_config:
#     #     logger.info("🌐 Not Initializing")
#     #     llm_config = LLM.objects.first()

#     # # Default to initialized model_with_tools if no config found or name is unknown
#     # if not llm_config:
#     #     logger.info("🌐 Not Initializing")
#     #     return model

#     # name = llm_config.name.lower()
#     if name == "gemini":
#         api_key = os.getenv("GEMINI_API_KEY")  # Always from env
#         model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
#         # Instantiate Gemini
#         # Standard safety settings can be added here as needed
#         llm_instance = ChatGoogleGenerativeAI(
#             model=model_name,
#             google_api_key=api_key,
#             temperature=0,
#             convert_system_message_to_human=True 
#         )
#         logger.info("🌐 Initializing Gemini")
#         # return llm_instance.bind_tools(tools)
#         return llm_instance 
#     elif name == "ollama_cloud":
#         # Use ollama_cloud as a special sentinel value
#         logger.info("🌐 Initializing Ollama Cloud LLM instance")
#         llm_instance = OllamaService(
#             base_url=os.getenv("OLLAMA_API_URL", "https://ai.notchhr.io/api/chat/local"),  
#             username=os.getenv("OLLAMA_USERNAME", "ai-user"),  
#             password=os.getenv("OLLAMA_PASSWORD", "x2GS7jEF@#2T"),  
#             model="ollama_cloud"  # Special sentinel value triggers cloud API
#         )
#         logger.info("🌐 Initilzized ollama_cloud ")
#         # return llm_instance.bind_tools(tools)
#         return llm_instance 
    
#     logger.info("🌐 Fall Back to Defaul Model config")
#     return model


# def get_model():
#     """
#     Lazy-loads the model and binds tools only when needed.
#     """
#     global _llm
    
#     if _llm is not None:
#         return _llm

#     try:
#         base_llm = get_llm_instance()
        
#         if base_llm is not None:
#             # 'tools' must be defined earlier in your file or imported
#             # llm= base_llm
#             logger.info("✅ Model and tools initialized successfully.")
#             return base_llm
        
#     except Exception as e:
#         logger.error(f"❌ Unexpected error in get_model_with_tools: {e}", exc_info=True)
    
#     return None


# # Graph Nodes
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
    do not hallucinate, either use the tool or response that you are not sure if unsure 

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

        PROTOCOL 4: DATA ANALYTICS AND VISUALIZATION
        - Use 'sql_query_tool' for data inquiries. Provide actionable insights.
        - For visualization requests (plot, chart, graph, visualize), ALWAYS chain: First call 'sql_query_tool' to fetch data, then call 'generate_visualization_tool' with the exact 'data' payload from 'sql_query_tool'.
        - IMPORTANT: When visualizing, you MUST call 'sql_query_tool' first to fetch the data. Once 'sql_query_tool' returns the JSON data, pass that exact 'data' payload into the `data` parameter of 'generate_visualization_tool'. Do not pass the data as a string, pass the raw data object.
        - If 'sql_query_tool' returns no data, do not call 'generate_visualization_tool'.

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
        - Visualization Analysis: {visualization_analysis}
        

        Tool Guide:{tool_intent_map}
        - **`generate_visualization_tool`**: **Use this tool when the user asks to 'plot', 'chart', 'graph', or 'visualize' data. You MUST supply the `data` parameter using the results from `sql_query_tool`.**
        - for sql related queries: Maximum of 3 follow up questions allowed, after which you must provide an answer based on the data or respond that you are unsure if the data is inconclusive. Do not call `generate_visualization_tool` if `sql_query_tool` returns no data or inconclusive results.
           
        ### Available Tools & Required Arguments:
        {tool_descriptions}

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

tool_guide = {
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
    "data_visualization": {
        "tools": ["sql_query_tool", "generate_visualization_tool"],
        "triggers": ["plot", "chart", "graph", "visualize", "show as a bar chart", "report", "count", "average", "total", "statistics", "data"]
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
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
            raise ValueError("No API key found for embeddings. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            logger.info("✅ Embeddings model initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Embeddings: {e}")
            raise
    return embeddings



 

import os
import shutil
import logging

def ingest_pdf_for_tenant(tenant_id: str, file_path: str):
    """
    Load a PDF, split into chunks, create a FAISS index.
    If an index already exists for the tenant, it is removed and replaced.
    """
    tenant_id = str(tenant_id)
    persist_directory = os.path.join("faiss_dbs", tenant_id)
    
    logger.info(f"[VectorStore | Tenant: {tenant_id}] Starting ingestion for {file_path}")
    
    try:
        # 1. Clean up existing index (Remove and Ingest logic)
        if os.path.exists(persist_directory):
            logger.info(f"[VectorStore | Tenant: {tenant_id}] Existing index found. Removing {persist_directory}...")
            shutil.rmtree(persist_directory)
        
        # 2. Path Validation
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
            
        # 3. Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Loaded {len(documents)} pages.")
        
        # 4. Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Split into {len(chunks)} chunks.")
        
        # 5. Embed and Create FAISS
        emb = get_embeddings()
        vector_store = FAISS.from_documents(chunks, emb)
        
        # 6. Save to Disk (Fresh directory)
        os.makedirs(persist_directory, exist_ok=True)
        vector_store.save_local(persist_directory)
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Successfully replaced index at {persist_directory}")
        
        return {"status": "success", "chunk_count": len(chunks), "path": persist_directory}
        
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Ingestion failed: {e}")
        return {"status": "error", "message": str(e)}

def ingest_pdf_for_tenantv1(tenant_id: str, file_path: str):
    """
    Load a PDF, split into chunks, create a FAISS index, and save to disk.
    """
    tenant_id = str(tenant_id)
    persist_directory = os.path.join("faiss_dbs", tenant_id)
    
    logger.info(f"[VectorStore | Tenant: {tenant_id}] Starting ingestion for {file_path}")
    
    try:
        # 1. Load PDF
        abs_path = os.path.abspath(file_path)
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Checking path: {file_path} (Absolute: {abs_path})")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Loaded {len(documents)} pages.")
        
        # 2. Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Split into {len(chunks)} chunks.")
        
        # 3. Embed and Create FAISS
        emb = get_embeddings()
        vector_store = FAISS.from_documents(chunks, emb)
        
        # 4. Save to Disk
        os.makedirs(persist_directory, exist_ok=True)
        vector_store.save_local(persist_directory)
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Saved index to {persist_directory}")
        
        return {"status": "success", "chunk_count": len(chunks), "path": persist_directory}
        
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Ingestion failed: {e}")
        return {"status": "error", "message": str(e)}


def initialize_vector_store(tenant_id: str):
    """
    Fetch and return a FAISS vector store for the given tenant.
    Loads ONLY from disk cache (faiss_dbs/<tenant_id>/).
    """
    tenant_id = str(tenant_id)
    persist_directory = os.path.join("faiss_dbs", tenant_id)

    logger.info(f"[VectorStore | Tenant: {tenant_id}] Initializing from disk (read-only).")

    # --- Step 1: Embeddings ---
    try:
        emb = get_embeddings()
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Embeddings ready.")
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Embeddings init failed: {e}")
        return None, {"status": "error", "message": f"Embeddings init failed: {e}"}

    # --- Step 2: Try to load from disk cache ---
    try:
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Directory {persist_directory} does not exist.")
            
        vector_store = FAISS.load_local(persist_directory, emb, allow_dangerous_deserialization=True)
        doc_count = vector_store.index.ntotal if hasattr(vector_store, "index") else "unknown"
        logger.info(f"[VectorStore | Tenant: {tenant_id}] Loaded successfully. index_size={doc_count}")
        return vector_store, {"status": "success", "source": "disk", "doc_count": doc_count}
    except FileNotFoundError:
        logger.warning(
            f"[VectorStore | Tenant: {tenant_id}] No disk index found at {persist_directory}. "
            "Use /load_pdf to ingest documents for this tenant."
        )
    except Exception as e:
        logger.error(f"[VectorStore | Tenant: {tenant_id}] Failed to load local index: {e}")

    return None, {"status": "not_found", "message": "No vector store index found on disk."}


# Model/Service Name Variables - Moved to llm_handler.py
# OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
_llm = None


GLOBAL_SCOPE = "GLOBAL"
NO_CONVO = "N/A"
def should_continue(state: State) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    if state is None:
        return END
    tenant_config = state.get("tenant_config", {}) or {}
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    log_info("Evaluating whether to continue or stop.", tenant_id, conversation_id)

    messages = state.get("messages", [])
    if not messages:
        return END
    # last_message = messages[-1]
    # FIX: Find the last AIMessage specifically
    last_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    log_debug(f"Last LLM message: {last_message}", tenant_id, conversation_id)

    if last_message and getattr(last_message, "tool_calls", None):
        log_info(f"LLM made tool calls: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}. Continuing to tool node.", tenant_id, conversation_id)
        return "tool_node"
    
    if last_message and "parsed_json" in last_message.additional_kwargs:
        parsed = last_message.additional_kwargs["parsed_json"]
    else:
        # Try parsing from content if missing
        import json
        try:
            if last_message:
                content_str = last_message.content if isinstance(last_message.content, str) else str(getattr(last_message, "content", ""))
                parsed = json.loads(content_str)
            else:
                parsed = {}
        except:
            parsed = {}

    tool_name = parsed.get("tool", "none")
    if not last_message or str(tool_name).lower() == "none" or not tool_name:
        log_info("LLM requested 'none' or no valid tool. Ending workflow.", tenant_id, conversation_id)
        return END

    log_info(
        f"LLM requested tool: {tool_name}. Continuing to tool node.", tenant_id, conversation_id
    )
    return "tool_node"

# Graph Nodes
def tool_node(state: State) -> dict:
    # Use the 'tools' list imported from tools.py
    from tools import tools as tool_list
    tools_by_name = {t.name: t for t in tool_list}
    
    tenant_config = state.get("tenant_config", {})
    tenant_id = tenant_config.get("tenant_id", "unknown")
    conversation_id = state.get("conversation_id", "unknown")
    
    log_info("Entered tool_node for execution.", tenant_id, conversation_id)
    
    try:
        last_ai_message = state["messages"][-1]
        tool_calls = getattr(last_ai_message, "tool_calls", [])
        log_debug(f"Tool calls to execute: {tool_calls}", tenant_id, conversation_id)
        
        if not tool_calls:
            return {"messages": []}

        new_messages = []
        state_updates = {}
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            # Handle nested 'arguments' key if present (LLM sometimes wraps args)
            if "arguments" in tool_args:
                tool_args = tool_args["arguments"]
            
            # Inject context
            tool_args["state"] = {k: v for k, v in state.items() if k != "messages"}
            
            tool = tools_by_name.get(tool_name)
            if not tool:
                log_error(f"Tool {tool_name} not found in {list(tools_by_name.keys())}", tenant_id, conversation_id)
                new_messages.append(ToolMessage(
                    content=f"Error: Tool {tool_name} not found.",
                    tool_call_id=tool_call.get("id"),
                    name=tool_name
                ))
                continue

            # Execution
            observation = tool.invoke(tool_args)
            
            # State Update Logic
            if tool_name == "pdf_retrieval_tool":
                state_updates["pdf_content"] = observation.get("pdf_content")
            
            if tool_name == "generate_visualization_tool":
                if isinstance(observation, dict) and "visualization_result" in observation:
                    viz_result = observation["visualization_result"]
                    state_updates["visualization_image"] = viz_result.get("image_base64")
                    state_updates["visualization_analysis"] = viz_result.get("analysis")
            
            new_messages.append(ToolMessage(
                content=json.dumps(observation) if isinstance(observation, dict) else str(observation),
                tool_call_id=tool_call.get("id"),
                name=tool_name
            ))
            
        log_info(f"Tool node executed successfully.", tenant_id, conversation_id)
        return {"messages": new_messages, **state_updates}

    except Exception as e:
        log_error(f"Critical failure in tool_node: {str(e)}", tenant_id, conversation_id)
        import traceback
        log_error(traceback.format_exc(), tenant_id, conversation_id)
        return {"messages": [ToolMessage(content=f"Error executing tool: {e}", tool_call_id="error", name="error_handler")]}


def extract_final_answer(response):
    """
    Robustly extract the final text answer from an LLM response or structured object.
    """
    try:
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
        except json.JSONDecodeError:
            pass

        # Try matching the largest possible JSON object first (greedy)
        greedy_match = re.search(r"(\{.*\})", content_clean, re.DOTALL)
        if greedy_match:
            try:
                obj = json.loads(greedy_match.group(1))
                if isinstance(obj, dict) and "answer" in obj:
                    return obj["answer"]
            except json.JSONDecodeError:
                pass

        # Try searching for JSON-like blocks as a fallback (non-greedy)
        json_blocks = re.findall(r"(\{.*?\})", content_clean, flags=re.DOTALL)
        for block in json_blocks:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "answer" in obj:
                    return obj["answer"]
            except json.JSONDecodeError:
                continue

        return content_clean or "LLM returned empty response"
    except Exception as e:
        logger.error(f"Error in extract_final_answer: {e}")
        return "Error extracting answer"

def clean_message_history(messages):
    """
    Strips additional_kwargs and serialized state from messages 
    to prevent LLM confusion and circular reference errors.
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            # Keep only the content and tool_calls, discard the state-heavy additional_kwargs
            cleaned.append(AIMessage(
                content=msg.content, 
                tool_calls=msg.tool_calls
            ))
        elif isinstance(msg, ToolMessage):
            # Keep only the tool result
            cleaned.append(ToolMessage(
                content=msg.content, 
                tool_call_id=msg.tool_call_id, 
                name=msg.name
            ))
        else:
            # HumanMessage and SystemMessage are usually safe
            cleaned.append(msg)
    return cleaned

def normalize_tool_calls(response):
    logger.info("Starting normalization of tool calls.")
    
    # 1. Check if tool calls already exist
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info("Tool calls already present in response.")
        return response

    if not isinstance(response.content, str):
        logger.info("Response content is not a string, skipping.")
        return response

    content = response.content.strip()
    if not content:
        logger.info("Response content is empty.")
        return response

    tool_calls = []
    decoder = json.JSONDecoder()
    pos = 0
    
    # 2. Iterate through content to find and parse multiple JSON objects
    # Attempt a regex extraction first to handle common LLM formatting issues like trailing quotes
    import re
    # Match everything between the first '{' and last '}'
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        content = match.group(1)

    while pos < len(content):
        try:
            # Find the next potential JSON object
            pos = content.find('{', pos)
            if pos == -1:
                break
            
            obj, index = decoder.raw_decode(content[pos:])
            pos += index
            
            # Check if this object is a tool call
            if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                logger.info(f"Successfully extracted tool call: {obj.get('tool') or obj.get('name')}")
                
                # Extract args, falling back to all other keys if not explicitly nested
                extracted_args = obj.get("parameters") or obj.get("args")
                if not extracted_args or not isinstance(extracted_args, dict):
                    extracted_args = {k: v for k, v in obj.items() if k not in ("tool", "name")}
                elif not extracted_args:
                    extracted_args = {}

                tool_calls.append({
                    "name": obj.get("tool") or obj.get("name"),
                    "args": extracted_args,
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call"
                })
            elif isinstance(obj, dict) and "visualization" in obj and isinstance(obj["visualization"], dict):
                 viz = obj["visualization"]
                 v_name = viz.get("tool") or viz.get("name")
                 if v_name:
                    logger.info(f"Successfully extracted tool call from visualization key: {v_name}")
                    v_args = viz.get("tool_input") or viz.get("args") or viz.get("parameters")
                    if not v_args or not isinstance(v_args, dict):
                        v_args = {k: v for k, v in viz.items() if k not in ("tool", "name")}
                    
                    tool_calls.append({
                        "name": v_name,
                        "args": v_args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    })
            elif isinstance(obj, dict) and "tool_call" in obj and isinstance(obj["tool_call"], dict):
                logger.info(f"Successfully extracted tool_call: {obj['tool_call'].get('name')}")
                tc = obj["tool_call"]
                args = tc.get("arguments") or tc.get("args", {})
                tool_calls.append({
                    "name": tc["name"],
                    "args": args,
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                    "type": tc.get("type", "tool_call")
                })
            elif isinstance(obj, dict) and "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                logger.info(f"Successfully extracted tool_calls array with {len(obj['tool_calls'])} calls")
                for tc in obj["tool_calls"]:
                    if isinstance(tc, dict) and "name" in tc:
                        # Map 'arguments' to 'args' if present
                        args = tc.get("arguments") or tc.get("args", {})
                        tool_calls.append({
                            "name": tc["name"],
                            "args": args,
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                            "type": tc.get("type", "tool_call")
                        })
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON object at position {pos}: {e}")
            pos += 1  # Skip to next character to attempt recovery
        except Exception as e:
            logger.error(f"Unexpected error during tool call normalization: {e}")
            break
            
    # 3. Apply normalized tool calls to response
    if tool_calls:
        response.tool_calls = tool_calls
        response.content = ""  # Clear raw text to clean up the UI
        logger.info(f"Normalization complete. {len(tool_calls)} tool calls detected.")
    else:
        logger.info("No tool calls detected in content.")
        
    return response

def assistant_node(state: State, config: RunnableConfig):
    """
    Consolidated Assistant Node: HR Support, Data Analytics, and Customer Concierge.
    Handles multiple roles and tool calling.
    """
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    conversation_id = config["configurable"].get("thread_id", "unknown")
    messages = state["messages"]
    logger.info(f"Messages before assistant processing Aliue: {messages}")
    log_info(f"Assistant node triggered for tenant: {tenant_id} with nessage : {messages}", tenant_id, conversation_id)

    # --- Resolve DB-sourced prompts with hardcoded fallbacks ---
    tenant_config = state.get("tenant_config") or {}
    # employee_id = {state.get('employee_id')}
    employee_id = str(state.get('employee_id', 'unknown'))
    current_year = datetime.now().year
    previous_year = current_year - 1
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    status_summary = state.get("status_summary", "No active application.")
    pdf_content = state.get("pdf_content", "None")
    web_content = state.get("web_content", "None")
    sql_result = state.get("sql_result", "None")
    visualization_analysis = state.get("visualization_analysis", "No visualization generated yet.")
    tool_intent_map = ((state.get("tenant_config") or {}).get("tool_intent_map")
  or tool_guide
    )

    push_name = tenant_config.get("push_name", "User")
    # global_answer_prompt acts as the persona/intro section of system_prompt.
    # Falls back to GLOBAL_FINAL_ANSWER_PROMPT if not set in the DB.
    global_answer_prompt = (
        tenant_config.get("global_answer_prompt")
        or GLOBAL_FINAL_ANSWER_PROMPT
    )
    
    # agent_prompt is the full system prompt stored in DB (used as an alternative
    # intro when set). If absent, the composed prompt below is used instead.
    # agent_prompt = tenant_config.get("agent_prompt") or None

    # Fetch the prompt template
    agent_prompt = tenant_config.get("agent_prompt",GLOBAL_FINAL_ANSWER_PROMPT)




    # Build string of tool descriptions and arguments
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}\n  Arguments schema: {t.args}" for t in tools])

    if agent_prompt:
        try:
            
         system_prompt = agent_prompt.format(
            ID=employee_id,
            current_year=current_year,
            previous_year=previous_year,
            current_date_str=current_date_str,
            pdf_content=pdf_content,
            web_content=web_content,
            # Using leave_application for the state result as requested
            sql_result=sql_result,
            visualization_analysis=visualization_analysis,
            status_summary=status_summary,
            tool_intent_map=tool_intent_map,
            tool_descriptions=tool_descriptions
        )
        except KeyError as e:
            logger.error(f"Missing key in prompt template: {e}")
            system_prompt = agent_prompt # Fallback to raw if format fails
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

    greeting_instruction = f"Always greet or address the user using their name: {push_name}, if they are starting a conversation or if appropriate."


    # 3. LLM INVOCATION
 
    # Check if last message was a tool result (we might already be done)
    if messages and isinstance(messages[-1], ToolMessage):
        logger.info("Last message was a tool result. LLM will now generate the final JSON response based on the tool's output.")

    # logger.info(f"!!! TRACE: assistant_node starting. Messages: {len(messages)}", flush=True)

    logger.info(f"!!! TRACE: assistant_node starting. Messages: {len(messages)}")
    try:
        # 1. Capture the raw response WITH bind_tools
        # Use the global 'llm' instance directly to avoid redundant/unstable DB calls inside the graph
        logger.info(f"Messages before assistant processing: {messages}")
        llm = get_model() 
        log_info(f"LLM instance: {llm}", tenant_id, conversation_id)
        if not llm:
            log_error("LLM instance is not available. Returning error response.", tenant_id, conversation_id)
            return {"messages": [AIMessage(content=json.dumps({"tool": "none", "answer": "Error: LLM not available."}))]}
        log_info(f"Tools: {tools}", tenant_id, conversation_id)
        llm_with_tools = llm.bind_tools(tools)
        log_info(f"LLM with tools: {llm_with_tools}", tenant_id, conversation_id)
        
        safe_messages = clean_message_history(state["messages"])
        log_info(f"Safe messages: {safe_messages}", tenant_id, conversation_id)
        try:
            response = llm_with_tools.invoke([SystemMessage(content=f"{system_prompt}\n\n{greeting_instruction}")] + safe_messages)
            log_info(f"LLM Raw Output: {response}", tenant_id, conversation_id)
        except Exception as e:
            log_error(f"LLM invoke failed: {e}", tenant_id, conversation_id)
            return {"messages": [AIMessage(content=json.dumps({"answer": "I'm sorry, I'm experiencing connectivity issues. Please try again later."}))]}
        
        log_info(f"LLM Raw Output: {response}", tenant_id, conversation_id)
    except BaseException as e:
        import traceback
        log_error(f"CRITICAL CRASH in assistant_node: {e}\n{traceback.format_exc()}", tenant_id, conversation_id)
        raise e
    refined_response = normalize_tool_calls(response)
    if refined_response.tool_calls:
        logger.info(f"Tool call detected: {refined_response.tool_calls}")
        return {"messages": [refined_response]}
    else:
        # Handle as standard text response
        final_answer = extract_final_answer(refined_response)
        logger.info(f"LLM response Assitant Node: {final_answer}")
        return {"messages": [AIMessage(content=final_answer)]}
    
    # if hasattr(refined_response, "tool_calls") and refined_response.tool_calls:
    #             logger.info(f"Tool calls foundAtejjd: {refined_response.tool_calls}")
    #             return {"messages": [refined_response]}  # keep the AIMessage intact

    # else:
    #     extract_final_answer(refined_response) and (not hasattr(refined_response, "tool_calls") or not refined_response.tool_calls):
    #         # Attach chart if present in state
    #     viz_result = state.get("visualization_result")
    #     if viz_result and "image_base64" in viz_result:
    #             refined_response.chart_base64 = viz_result["image_base64"]
                
    #         return {
    #             "messages": [AIMessage(content=refined_response.content)],
    #             "status_summary": status_summary
    #         }   

def build_graph(tenant_id: str, conversation_id: str, checkpointer=None):
    workflow = StateGraph(State)
    log_info("Building graph for tenant: {tenant_id}, conversation: {conversation_id}", tenant_id, conversation_id)

    # 1. Add Nodes
    workflow.add_node("assistant_node", assistant_node)
    workflow.add_node("tool_node", tool_node)
    
    
    # workflow.add_node("guardrail_node", routing_guardrail_node)
    # log_info("Guardrail node added to graph.", tenant_id, conversation_id)

    # 2. Routing
    workflow.add_edge(START, "assistant_node")
    log_info("Edge added from START to assistant_node.", tenant_id, conversation_id)

    # workflow.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    workflow.add_conditional_edges(
        "assistant_node", should_continue, ["tool_node", END]
    )

   # After tool execution, always return to assistant
    workflow.add_edge("tool_node", "assistant_node")
    log_info("Edge added from tool_node to assistant_node.", tenant_id, conversation_id)

    return workflow.compile(checkpointer=checkpointer)



def process_message(message_content: str, conversation_id: str, tenant_id: str, employee_id: Optional[str] = None, push_name: str = "User"):
    # Fallback for employee_id
    if not employee_id:
        employee_id = DEFAULT_EMPLOYEE_ID
    log_info("Starting message processing pipeline", tenant_id, conversation_id)
    log_info(f" alukett Employee ID: {employee_id}-message Content: {message_content}", tenant_id, conversation_id)
    
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
        
    prompt_id = None
    db_uri = os.getenv("DATABASE_URL")
    p_res = None

    try:
        with SessionLocal() as session:
            log_info("Fetching tenant AI config", tenant_id, conversation_id)
            # 1. Fetch Tenant AI config
            ta_sql = """
                SELECT prompt_template_id, db_uri 
                FROM customer_tenant_ai 
                WHERE tenant_id = (SELECT id FROM org_tenant WHERE code = :code)
            """
            ta_item = session.execute(text(ta_sql), {"code": tenant_id}).fetchone()
            
            prompt_id = ta_item[0] if ta_item else None
            db_uri = ta_item[1] if ta_item and ta_item[1] else db_uri
            
            # 2. Fetch Prompt
            if prompt_id:
                p_sql = "SELECT agent_prompt, \"global_answer_prompt\", \"tool_intent_map\" FROM customer_prompt WHERE id = :pid"
                p_res = session.execute(text(p_sql), {"pid": prompt_id}).fetchone()
            else:
                p_sql = "SELECT agent_prompt, \"global_answer_prompt\", \"tool_intent_map\" FROM customer_prompt WHERE name = 'standard' LIMIT 1"
                p_res = session.execute(text(p_sql)).fetchone()
            log_info("Fetched tenant AI config", tenant_id, conversation_id)    
    except Exception as e:
        log_error(f"Database configuration fetch failed: {e}. Using defaults.", tenant_id, conversation_id)
    
    if p_res:
        log_info(f"Prompt template found: {p_res[0][:50]}...", tenant_id, conversation_id)
        system_prompt = p_res[0]
    else:
        log_warning("No prompt template found in DB. Using global default.", tenant_id, conversation_id)
        system_prompt = GLOBAL_FINAL_ANSWER_PROMPT    
  
    tenant_config_dict = {
        "tenant_id": tenant_id,
        "employee_id": employee_id,
        "db_uri": db_uri,
        "push_name": push_name,
        "agent_prompt": p_res[0] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
        "final_answer_prompt": p_res[1] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
        "tool_intent_map": p_res[2] if p_res else tool_guide,
    }
    # --- Logging Connectivity confirmation ---
    # 1. Database Connectivity Check
    db_connected = False
    record_found = False
    try:
        with SessionLocal() as check_session:
            # Check for a simple record to confirm DB connectivity and data presence
            res = check_session.execute(text("SELECT id, name, code FROM org_tenant WHERE code = :code LIMIT 1"), {"code": tenant_id}).fetchone()
            log_info(f"✅ DB Connected. Tenant record found: ID={res[0]}, Name='{res[1]}', Code='{res[2]}'.", tenant_id, conversation_id)
            db_connected = True
            if res:
                record_found = True
                log_info(f"✅ DB Connected. Tenant record found: ID={res[0]}, Name='{res[1]}', Code='{res[2]}'.", tenant_id, conversation_id)
                # Fetch a sample record from another table to be double sure - e.g. Prompt or LeaveType
                sample = check_session.execute(text("SELECT id, name FROM customer_prompt LIMIT 1")).fetchone()
                if sample:
                    log_info(f"📊 Sample Prompt Record: ID={sample[0]}, Name='{sample[1]}'", tenant_id, conversation_id)
            else:
                log_warning(f"⚠️ DB Connected, but NO tenant record found for '{tenant_id}'.", tenant_id, conversation_id)
    except Exception as e:
        log_error(f"❌ DB Connectivity check failed: {e}", tenant_id, conversation_id)

    # 2. FAS URL Connectivity Check
    # fas_url = os.getenv("_FAS_REMOTE_URL") or os.getenv("VECTOR_STORE_URL")
# 1. Define the base path using the current script's location
# This ensures it always looks in the correct 'faiss_dbs' folder regardless of where you run it from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(base_dir, "faiss_dbs", tenant_id)

    # 2. Check the directory using OS-native methods
    try:
        if os.path.isdir(persist_directory):
            log_info(f"✅ FAS Directory confirmed at: {persist_directory}", tenant_id, conversation_id)
            
            # Check for the existence of the actual index files to be extra safe
            index_file = os.path.join(persist_directory, "index.faiss")
            if os.path.exists(index_file):
                log_info(f"✅ Index file found: {index_file}", tenant_id, conversation_id)
            else:
                log_error(f"⚠️ FAS Directory exists but index.faiss is missing: {persist_directory}", tenant_id, conversation_id)
        else:
            log_error(f"❌ FAS Directory not found at: {persist_directory}", tenant_id, conversation_id)

    except Exception as e:
        log_error(f"❌ Unexpected error accessing directory {persist_directory}: {e}", tenant_id, conversation_id)
        
        
    log_info(f"Tenant config prepared. DB URI present: {bool(db_uri)}", tenant_id, conversation_id)
    
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    # with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as cp:
        checkpointer.setup()
        # app = build_graph(tenant_id, conversation_id,checkpointer=checkpointer)
        

        # config = {"configurable": {"thread_id": conversation_id, "tenant_id": tenant_id, "employee_id": employee_id}}
        
        config = {"configurable": {"thread_id": conversation_id}}

        # initial_state = {
        #     "messages": [HumanMessage(content=message_content)],
        #     "user_query": message_content,
        #     "tenant_config": tenant_config_dict,
        #     "vector_store_path": persist_directory
        # }
        
      
        context = Context(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            emp_id=employee_id,
            db_uri=db_uri,
            push_name=push_name,
            agent_prompt=p_res[0] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
            final_answer_prompt=p_res[1] if p_res else GLOBAL_FINAL_ANSWER_PROMPT,
            tool_intent_map=p_res[2] if p_res else tool_guide,
            vector_store_path=persist_directory
        )
              
        agent = create_agent(
    model=get_model(),
    system_prompt=system_prompt,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

        # final_state = app.invoke(initial_state, config=config)
        # final_state = agent.invoke({"messages": [HumanMessage(content=message_content)]}, config=config)
        final_state = agent.invoke(
    {"messages": [{"role": "user", "content": message_content}]},
    config=config,
    context=context
)
        logger.info(f" Raw LLM response  : {final_state}")
        
        # Correctly access the final answer from the messages in the state
        messages = final_state.get("messages", [])
        # if messages:
        current_answer = messages[-1].content if messages else ""
            
        try:
            logger.info(f"LLM response Process message : {current_answer}")
            
            # 1. Extraction Logic: Handle objects (ResponseFormat), dictionaries, or JSON strings
            if current_answer:
                # If it's the ResponseFormat object
                if hasattr(current_answer, 'answer'):
                    extracted_text = current_answer.answer
                elif hasattr(current_answer, 'punny_response'):
                    extracted_text = current_answer.punny_response
                # If it's already a dictionary
                elif isinstance(current_answer, dict):
                    extracted_text = current_answer.get("answer", current_answer.get("punny_response", str(current_answer)))
                # If it's a string, try to parse as JSON in case it's a JSON-formatted string
                elif isinstance(current_answer, str):
                    try:
                        # Clean up common AI markdown wrappers
                        cleaned_str = re.sub(r"```json\s*|\s*```", "", current_answer.strip())
                        data = json.loads(cleaned_str)
                        if isinstance(data, dict):
                            extracted_text = data.get("answer", data.get("punny_response", current_answer))
                        else:
                            extracted_text = current_answer
                    except (json.JSONDecodeError, AttributeError):
                        extracted_text = current_answer
                else:
                    extracted_text = str(current_answer)
                    
                # 2. Build the result dictionary
                # result = {"answer": extracted_text}
                result = extracted_text
                # Add visualization fields if present in final_state
                # if final_state.get("visualization_image"):
                #     result["visualization_image"] = final_state["visualization_image"]
                # if final_state.get("visualization_analysis"):
                #     result["visualization_analysis"] = final_state["visualization_analysis"]
                    
                logger.info(f"LLM Response Extracted: {extracted_text}")
                return extracted_text

            else:
                fallback = "I apologize, but I encountered an internal error processing your request."
                logger.info(f"LLM Response Fallback: {fallback}")
                return {"answer": fallback}

        except Exception as e:
            logger.error(f"Error extracting final response: {str(e)}", exc_info=True)
            return "An unexpected error occurred while formatting the response."