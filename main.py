import os
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base
from .chat_bot import process_message

# Create DB tables on startup (if they don't exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chatbot API", description="FastAPI Refactor of the Chatbot application")

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    tenant_id: str
    employee_id: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Chatbot API is running"}

@app.post("/webhook")
async def chatbot_webhook(request: ChatRequest):
    """
    Main endpoint for receiving chat messages. 
    Relays to process_message in chat_bot.py.
    """
    try:
        response = process_message(
            message_content=request.message,
            conversation_id=request.conversation_id,
            tenant_id=request.tenant_id,
            employee_id=request.employee_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
