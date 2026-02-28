import os
import logging
import httpx
import redis
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pydantic import BaseModel
from ollama import Client

load_dotenv()

# --- 1. Driver & URL Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Force the psycopg (v3) driver to match your requirements.txt
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

REDIS_URL = os.getenv("REDIS_URL")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL")

# --- 2. Database & Redis Setup ---
engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

redis_client = redis.Redis.from_url(REDIS_URL)
ollama_client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)
# loghger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- 3. Database Model ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("whatsapp_webhook")

class UserState(Base):
    __tablename__ = "user_states"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True)
    state = Column(String(50)) 
    step = Column(String(50))
    temp_data = Column(Text)

# Create tables on startup
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"⚠️ Connection Error: {e}")

# --- 4. Pydantic Models ---
class WebhookPayload(BaseModel):
    phone_number: Optional[str] = None
    message: Optional[str] = None
    event: Optional[str] = None
    instance: Optional[str] = None
    data: Optional[dict] = None

# --- 5. Helper Functions ---
async def send_whatsapp_message(phone_number: str, text: str):
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {"apikey": EVOLUTION_API_KEY, "Content-Type": "application/json"}
    payload = {"number": phone_number, "text": text}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            return response.json()
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {e}")
            return None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 6. FastAPI Routes ---
app = FastAPI(title="ATB AI WhatsApp Integration")

@app.post("/webhook")
async def whatsapp_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    # Data Extraction
    logger.info(f"Received webhook: {payload}")
    if payload.data:
        logger.info("Extracting data from payload.data")
        data = payload.data
        msg_obj = data.get("message", {})
        message_content = msg_obj.get("conversation") or \
                          msg_obj.get("extendedTextMessage", {}).get("text")
        remote_jid = data.get("key", {}).get("remoteJid", "")
        phone_number = remote_jid.split("@")[0]
    else:
        logger.info("Extracting data from payload fields")
        phone_number = payload.phone_number
        message_content = payload.message

    if not message_content or not phone_number:
        logger.warning("Missing message content or phone number")
        return {"status": "ignored", "reason": "Missing data"}

    message = message_content.strip()

    # State Management
    user = db.query(UserState).filter(UserState.phone_number == phone_number).first()
    if not user:
        user = UserState(phone_number=phone_number, state="start", step="intro")
        db.add(user)
        db.commit()
        db.refresh(user)

    # State Transition: leave_application
    if any(word in message.lower() for word in ["leave", "permission", "sick"]):
        user.state = "leave_application"
        user.step = "asking_details"
        db.commit()

    # AI Response Logic
    system_prompt = f"You are an HR Assistant for ATB AI. Current User State: {user.state}."
    if user.state == "leave_application":
        system_prompt += " Be helpful. Ask the user for their leave reason and dates."

    try:
        response = ollama_client.chat(
            model=OLLAMA_CLOUD_MODEL,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': message}]
        )
        logging.info(f"AI response: {response}")
        response_text = response['message']['content']
    except Exception:
        response_text = "E pele, I'm having trouble connecting to my brain. Try again soon!"

    # Final Action
    await send_whatsapp_message(phone_number, response_text)
    logger.info(f"Sent response to {phone_number}: {response_text}")
    return {"status": "processed", "state": user.state}
    
@app.get("/")
async def root():
    logger.info("Health check endpoint hit")
    return {"status": "online", "database": "connected"}

@app.get("/utility/")
def utility_root():
    logger.info("Utility endpoint hit")
    return {"message": "Hello from ATB AI!"}