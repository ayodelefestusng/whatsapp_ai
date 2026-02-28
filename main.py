from fastapi import FastAPI, Request, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from dotenv import load_dotenv
import redis
from ollama import Client
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db") 
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") 
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from pydantic import BaseModel

class WebhookPayload(BaseModel):
    phone_number: str
    message: str



class UserState(Base):
    __tablename__ = "user_states"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True)
    state = Column(String(50))
    step = Column(String(50))
    temp_data = Column(Text)

Base.metadata.create_all(bind=engine)

# --- Redis Setup ---
REDIS_URL = os.getenv("REDIS_URL", "redis://default:65f11924ebc7c9e25051@whatsapp-1_evolution-api-redis:6379")
redis_client = redis.Redis.from_url(REDIS_URL)
# Ollama client setup 

client = Client(
    host='https://ollama.com',
headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)



# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "online", "database": "connected"}

@app.get("/utility/")
def read_root():
    return {"message": "Hello from ATB AI!"}

@app.get("/ayodeles/")
def read_roots():
    return {"message": "Atm ATB AI!"}
# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/webhook")
async def whatsapp_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    phone_number = payload.phone_number
    message = payload.message

    # Store state in Postgres
    user = db.query(UserState).filter(UserState.phone_number == phone_number).first()
    if not user:
        user = UserState(phone_number=phone_number, state="new", step="start", temp_data="")
        db.add(user)
        db.commit()
        db.refresh(user)

    # Cache last message in Redis
    redis_client.set(f"user:{phone_number}:last_message", message)

    # Generate AI response using Ollama Cloud
    messages = [{"role": "user", "content": message}]
    response_text = ""
    for part in client.chat(OLLAMA_CLOUD_MODEL, messages=messages, stream=True):
        response_text += part['message']['content']

    return {
        "status": "received",
        "phone_number": phone_number,
        "message": message,
        "ai_response": response_text
    }
