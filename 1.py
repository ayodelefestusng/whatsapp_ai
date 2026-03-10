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




class WebhookPayload(BaseModel):
    phone_number: str
    message: str

# --- 4. Dependencies ---
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# --- 5. FastAPI App & Routes ---
app = FastAPI(title="ATB AI WhatsApp Integration")

@app.get("/")
async def root():
    return {"status": "online", "database": "connected"}

@app.get("/utility/")
def utility_root():
    return {"message": "Hello from ATB AI!"}


async def send_whatsapp_message(phone_number: str, text: str):
    """Sends a message via Evolution API with corrected JID formatting."""
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Ensure the number has the @s.whatsapp.net suffix
    clean_number = phone_number.replace("+", "").strip()
    if "@" not in clean_number:
        recipient = f"{clean_number}@s.whatsapp.net"
    else:
        recipient = clean_number

    payload = {
        "number": recipient, # Use the formatted JID
        "text": text,
        "linkPreview": False # Good practice to disable if not needed
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # We catch the error and log it, as per your logging preference
            error_detail = e.response.text
            logger.error(f"Evolution API Error: {error_detail}")
            return {"error": str(e), "details": error_detail}
        
# --- 6. Helper Functions ---
async def send_whatsapp_message1(phone_number: str, text: str):
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
async def whatsapp_webhook(payload: WebhookPayload):
    # phone_number = payload.phone_number
    # message = payload.message
    message="Hello Can I get a loan"
    phone_number="2348021299221"

    logger.info(f"Incoming: {phone_number} -> {message}")

    # 1. State Management & State Retrieval
   
    # 3. Generate AI Response via Ollama
    # Construct a prompt based on the user's current state
    system_prompt = (
        "You are a helpful HR Assistant for ATB AI. "
        # f"The user's current state is: {user.state}. "
    )
    system_prompt += "Help the user complete their leave application. Ask for reason, start date, and end date if missing."

    # if user.state == "leave_application":
    #     system_prompt += "Help the user complete their leave application. Ask for reason, start date, and end date if missing."
    # else:
    #     system_prompt += "Answer general questions politely. Keep responses concise for WhatsApp."

    try:
        # Calling the Ollama Cloud Client
        response = ollama_client.chat(
            model=OLLAMA_CLOUD_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': message},
            ]
        )
        response_text = response['message']['content']
        logger.info(f"Final Response: {response_text} -> {message}")
    except Exception as e:
        logger.error(f"Ollama Error: {str(e)}")
        response_text = "I'm having a bit of trouble processing that right now. Could you try again in a moment?"

    # 4. Redis Caching (Store history for context if needed later)
    redis_client.set(f"user:{phone_number}:last_message", message, ex=3600)

    # 5. Evolution API Send
    send_result = await send_whatsapp_message(phone_number, response_text)

    return {
        "status": "processed",
        "state": "user.state",
        "ai_response": response_text,
        "evolution_result": send_result
    }

  
    
# Inside your whatsapp_webhook function:

# 1. State Management logic

