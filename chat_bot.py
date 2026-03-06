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
from typing import Any, Dict, List, Literal, Optional, Union, Annotated
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
from .database import SessionLocal, Tenant_AI, Conversation, Message, Prompt, LLM
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

# Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
GLOBAL_FINAL_ANSWER_PROMPT = """You are Damilola, the AI-powered virtual assistant. Deliver professional customer service."""
REMOTE_FAISS_URL = "http://147.182.194.8:3000/projects/whatsapp-1/app/vectra_app"

embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    return embeddings

def initialize_vector_store(tenant_id: str):
    """
    Handles FAISS vector store. 
    In FastAPI version, we attempt to download/load from the remote URL if needed.
    """
    persist_directory = os.path.join("faiss_dbs", tenant_id)
    # Check if local index exists, if not, we would normally fetch from REMOTE_FAISS_URL
    # For now, we'll look for local files as a fallback
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    
    emb = get_embeddings()
    try:
        vector_store = FAISS.load_local(persist_directory, emb, allow_dangerous_deserialization=True)
        return vector_store, {"status": "success"}
    except:
        # Create empty index if load fails
        vector_store = FAISS.from_texts([" "], emb)
        return vector_store, {"status": "empty"}

def get_llm_instance(tenant_id=None):
    """Fetches LLM config from DB using SQLAlchemy."""
    with SessionLocal() as session:
        llm_config = session.query(LLM).first()
        if not llm_config:
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        name = llm_config.name.lower()
        if name == "gemini":
            return ChatGoogleGenerativeAI(model=llm_config.model or "gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        elif "ollama" in name:
            return OllamaService(model=llm_config.model or "llama3")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Graph Nodes
def assistant_node(state: State, config: RunnableConfig):
    tenant_id = config["configurable"].get("tenant_id", "unknown")
    llm = get_llm_instance(tenant_id)
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
    system_msg = SystemMessage(content=GLOBAL_FINAL_ANSWER_PROMPT)
    
    response = llm_with_tools.invoke([system_msg] + messages)
    
    # Robust parsing for tool calls (Ollama fix)
    if isinstance(response.content, str) and not response.tool_calls:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                t_name = parsed.get("name") or parsed.get("tool")
                if t_name:
                    response.tool_calls = [{"name": t_name, "args": parsed.get("args", {}), "id": str(uuid.uuid4()), "type": "tool_call"}]
            except: pass

    return {"messages": [response]}

def tool_node(state: State):
    last_msg = state["messages"][-1]
    new_messages = []
    tools_by_name = {t.name: t for t in tools}
    
    for call in last_msg.tool_calls:
        tool = tools_by_name.get(call["name"])
        if tool:
            obs = tool.invoke({**call["args"], "state": state})
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

def process_message(message_content: str, conversation_id: str, tenant_id: str, employee_id: str):
    """FastAPI entry point for chat logic."""
    with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as cp:
        cp.setup()
        app = build_graph(cp)
        
        config = {"configurable": {"thread_id": conversation_id, "tenant_id": tenant_id, "employee_id": employee_id}}
        initial_state = {"messages": [HumanMessage(content=message_content)]}
        
        final_state = app.invoke(initial_state, config=config)
        answer = final_state["messages"][-1].content
        
        return {"answer": answer}