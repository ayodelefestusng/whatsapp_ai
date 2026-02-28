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

# --- 1. Load Environment & Configuration ---
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("whatsapp_webhook")

# Environment Variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

REDIS_URL = os.getenv("REDIS_URL", "redis://default:65f11924ebc7c9e25051@whatsapp-1_evolution-api-redis:6379")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "https://whatsapp-1-evolution-api.xqqhik.easypanel.host")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "429683C4C977415CAAFCCE10F7D57E11")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "session1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")

# --- 2. Database & Redis Setup ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

redis_client = redis.Redis.from_url(REDIS_URL)
ollama_client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)

# --- 3. Models ---
class UserState(Base):
    __tablename__ = "user_states"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True)
    state = Column(String(50)) # Use 'leave_application' here when active
    step = Column(String(50))
    temp_data = Column(Text)

Base.metadata.create_all(bind=engine)

class WebhookPayload(BaseModel):
    phone_number: str
    message: str

# --- 4. Dependencies ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 5. FastAPI App & Routes ---
app = FastAPI(title="ATB AI WhatsApp Integration")

@app.get("/")
async def root():
    return {"status": "online", "database": "connected"}

@app.get("/utility/")
def utility_root():
    return {"message": "Hello from ATB AI!"}

# --- 6. Helper Functions ---
async def send_whatsapp_message(phone_number: str, text: str):
    """Sends a message via Evolution API using verified connection settings."""
    # Note: We use the URL structure verified in your terminal test
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "number": phone_number,
        "text": text
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Error sending message: {e.response.text}")
            return {"error": str(e), "details": e.response.text}

# --- 7. Webhook Logic ---
@app.post("/webhook")
async def whatsapp_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    phone_number = payload.phone_number
    message = payload.message

    logger.info(f"Incoming: {phone_number} -> {message}")

    # 1. State Management
    user = db.query(UserState).filter(UserState.phone_number == phone_number).first()
    
    if not user:
        # Initialize new user
        user = UserState(phone_number=phone_number, state="start", step="intro", temp_data="")
        db.add(user)
    else:
        # Update existing state - Example: transition to leave_application
        user.state = "leave_application" 
        user.temp_data = message
    
    db.commit()
    db.refresh(user)

    # 2. Redis Caching
    redis_client.set(f"user:{phone_number}:last_message", message, ex=3600) # Expire in 1hr

    # 3. AI Response (Placeholder for Ollama Logic)
    response_text = f"Received your message for your {user.state}: {message}"

    # 4. Evolution API Send
    send_result = await send_whatsapp_message(phone_number, response_text)

    return {
        "status": "processed",
        "state": user.state,
        "evolution_result": send_result
    }