import os
import logging
import httpx
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pydantic import BaseModel
from ollama import Client

load_dotenv()

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
# Force the correct driver for SQLAlchemy 2.0 + Python 3.13
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Ensure Redis uses the Public IP if running locally
REDIS_URL = os.getenv("REDIS_URL") 

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL")

# --- Database & Redis Setup ---
# Added connect_args to help with timeout issues
engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

redis_client = redis.Redis.from_url(REDIS_URL)
ollama_client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)

# --- Models ---
class UserState(Base):
    __tablename__ = "user_states"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True)
    state = Column(String(50)) 
    step = Column(String(50))
    temp_data = Column(Text)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"⚠️ Could not create tables: {e}. Check if port 5432 is exposed!")

from typing import Optional

class WebhookPayload(BaseModel):
    # For your manual tests
    phone_number: Optional[str] = None
    message: Optional[str] = None
    
    # For the real Evolution API
    event: Optional[str] = None
    instance: Optional[str] = None
    data: Optional[dict] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="ATB AI WhatsApp Integration")
@app.get("/")
async def root():
    logging.info("Health check endpoint hit")
    return {"status": "online", "database": "connected"}

@app.get("/utility/")
def utility_root():
    logging.info("Utility endpoint hit")
    return {"message": "Hello from ATB AI!"}

# --- Webhook Logic ---
@app.post("/webhook")
async def whatsapp_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    # --- SMART DATA EXTRACTION ---
    if payload.data:
        # It's a real Evolution API request
        data = payload.data
        message_content = data.get("message", {}).get("conversation") or \
                          data.get("message", {}).get("extendedTextMessage", {}).get("text")
        
        remote_jid = data.get("key", {}).get("remoteJid", "")
        phone_number = remote_jid.split("@")[0]
    else:
        # It's your manual PowerShell/Postman test
        phone_number = payload.phone_number
        message_content = payload.message

    # Stop if we still don't have the basics
    if not message_content or not phone_number:
        logging.warning("Webhook received but couldn't find phone or message.")
        return {"status": "ignored", "reason": "Missing data"}

    message = message_content.strip()
    # ------------------------------

    logging.info(f"Processing: {phone_number} -> {message}")
    
    # ... Rest of your State Management (leave_application) and Ollama logic ...

    # 1. State Management
    user = db.query(UserState).filter(UserState.phone_number == phone_number).first()
    if not user:
        user = UserState(phone_number=phone_number, state="start", step="intro", temp_data="")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 2. State Transition: leave_application
    if any(word in message.lower() for word in ["leave", "permission", "sick"]):
        user.state = "leave_application"
        user.step = "asking_details"
        db.commit()

    # 3. AI Response
    system_prompt = f"You are an HR Assistant. User State: {user.state}. "
    if user.state == "leave_application":
        system_prompt += "Ask for the leave reason and the start/end dates."
    else:
        system_prompt += "Answer work questions concisely."

    try:
        response = ollama_client.chat(
            model=OLLAMA_CLOUD_MODEL,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': message}]
        )
        response_text = response['message']['content']
    except Exception:
        response_text = "E pele, I'm having trouble connecting to my brain. Try again soon!"

    # 4. Send Result
    await send_whatsapp_message(phone_number, response_text)

    return {"status": "processed", "state": user.state}

async def send_whatsapp_message(phone_number: str, text: str):
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {"apikey": EVOLUTION_API_KEY, "Content-Type": "application/json"}
    payload = {"number": phone_number, "text": text}
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=payload, headers=headers)