import os
import json
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from pydantic import BaseModel
from typing import Optional, Any

from tools import log_debug
from database import engine, Base
from chat_bot import log_info,log_error, process_message

# Ensure tables exist (legacy check, though FastAPI doesn't manage them)
# Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chatbot API", description="FastAPI Refactor with WhatsApp Integration")

# Hardcoded fallback as requested
DEFAULT_EMPLOYEE_ID = "obinna.kelechi.adewale@dignityconcept.tech"

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    tenant_id: Optional[str] = "DMC"
    employee_id: Optional[str] = DEFAULT_EMPLOYEE_ID
    pushName: Optional[str] = "User"

@app.get("/")
def read_root():
    return {"status": "online", "message": "Chatbot API is running"}

@app.post("/chatbot_webhook")
async def chatbot_webhook(request: ChatRequest):
    """
    Endpoint for Postman or internal testing.
    """
    try:
        response = process_message(
            message_content=request.message,
            conversation_id=request.conversation_id or "postman_session",
            tenant_id=request.tenant_id or "DMC",
            employee_id=request.employee_id or DEFAULT_EMPLOYEE_ID,
            push_name=request.pushName or "User"
        )
        return response
    except Exception as e:
        log_error(f"Error in chatbot_webhook: {e}", request.tenant_id or "DMC", "postman_session")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatsapp_webhook")
async def whatsapp_webhook(request: Request):
    """
    Standardizes WhatsApp integration (e.g. from Evolution API).
    Maps conversation_id to phone number.
    """
    log_info("Webhook url called ", "tenant_id", "conversation_id")

    try:
        # Check if it's JSON or Form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            payload = await request.json()
            # Extract basic info (Adapting to common Evolution API / WhatsApp webhook structures)
            # data.key.remoteJid usually contains the phone number
            # Evolution API example: payload['data']['key']['remoteJid']
            # Generic fallback: look for 'sender', 'from', or 'phone'
            
            message_text = ""
            phone_number = "unknown"
            push_name = "User"
            
            # Evolution API v2 structure
            if "data" in payload and isinstance(payload["data"], dict):
                data = payload["data"]
                phone_number = data.get("key", {}).get("remoteJid", "").split("@")[0]
                push_name = data.get("pushName") or "User"
                message_text = data.get("message", {}).get("conversation", "") or \
                               data.get("message", {}).get("extendedTextMessage", {}).get("text", "")
            
            # Generic/Fallback
            if not message_text:
                message_text = payload.get("message", {}).get("text") or payload.get("text", "")
            if phone_number == "unknown":
                phone_number = payload.get("sender") or payload.get("from") or "anonymous"
            if push_name == "User":
                push_name = payload.get("pushName") or "User"
            
            tenant_id = payload.get("tenant_id", "DMC")
            employee_id = payload.get("employee_id", DEFAULT_EMPLOYEE_ID)
            
        else:
            # Handle Form data (request.POST equivalent)
            form_data = await request.form()
            message_text = form_data.get("message", "")
            phone_number = form_data.get("phone_number") or form_data.get("sender") or "anonymous"
            push_name = form_data.get("pushName") or "User"
            tenant_id = form_data.get("tenant_id", "DMC")
            employee_id = form_data.get("employee_id", DEFAULT_EMPLOYEE_ID)
        
        log_info("Starting message processing pipeline", tenant_id, phone_number)

        if not message_text:
            return {"status": "ignored", "reason": "empty message"}

        response = process_message(
            message_content=message_text,
            conversation_id=phone_number, # Map to phone number as requested
            tenant_id=tenant_id,
            employee_id=employee_id,
            push_name=push_name
        )
        return response

    except Exception as e:
        log_error(f"Error in whatsapp_webhook: {e}", "unknown", "unknown")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
