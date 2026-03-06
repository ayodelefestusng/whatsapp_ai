import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Build SQLAlchemy-compatible URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/hris?sslmode=disable")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Tenant_AI(Base):
    __tablename__ = "customer_tenant_ai"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), unique=True, index=True)
    tenant_name = Column(String(255))
    vector_store_path = Column(String(500))
    chatbot_greeting = Column(Text)
    agent_node_prompt = Column(Text)
    final_answer_prompt = Column(Text)
    summary_prompt = Column(Text)
    db_uri = Column(String(500))
    tenant_website = Column(String(500))
    tenant_knowledge_base = Column(String(500))
    sentiment_threshold = Column(Integer, default=0)
    is_hum_agent_allow = Column(Boolean, default=True)
    conf_level = Column(Integer, default=40)
    ticket_type = Column(JSON) # List of strings
    message_tone = Column(String(100), default="Professional")

class Conversation(Base):
    __tablename__ = "customer_conversation"
    id = Column(String(100), primary_key=True)
    tenant_id = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = Column(Text)
    metadata = Column(JSON)

class Message(Base):
    __tablename__ = "customer_message"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), index=True)
    role = Column(String(50)) # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class Prompt(Base):
    __tablename__ = "customer_prompt"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), index=True)
    name = Column(String(100))
    content = Column(Text)

class LLM(Base):
    __tablename__ = "customer_llm"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), index=True)
    provider = Column(String(100)) # 'openai', 'gemini', 'ollama'
    model_name = Column(String(100))
    api_key = Column(String(255))
    base_url = Column(String(255))

class UserState(Base):
    __tablename__ = "user_states"
    phone_number = Column(String(50), primary_key=True, index=True)
    state = Column(String(50), default="idle")
    step = Column(String(50), nullable=True)
    data = Column(JSON, default={})
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables if they don't exist
# Base.metadata.create_all(bind=engine)