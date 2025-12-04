"""
Voice-Order Restaurant AI Agent - Main FastAPI Application
Handles incoming Twilio voice calls, transcription, LLM processing, and TTS responses.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, get_order_prompt
from utils.transcription import transcribe_audio
from utils.rag import MenuRAG
from utils.tts import text_to_speech, get_audio_url
from utils.session import SessionManager
from utils.database import Database
from utils.payment import PaymentProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('call_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
session_manager = SessionManager()
menu_rag = MenuRAG()
database = Database()
payment_processor = PaymentProcessor()

# Twilio client
twilio_client = None
if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
    twilio_client = TwilioClient(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("🚀 Voice AI Ordering Agent starting up...")
    menu_rag.initialize()
    database.initialize()
    logger.info("✅ All components initialized successfully")
    yield
    # Shutdown
    logger.info("👋 Voice AI Ordering Agent shutting down...")
    session_manager.cleanup_all()


app = FastAPI(
    title="Voice-Order Restaurant AI Agent",
    description="Conversational AI for restaurant ordering via phone calls",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Voice-Order Restaurant AI Agent",
        "version": "1.0.0",
        "endpoints": {
            "voice_webhook": "/voice",
            "sms_webhook": "/sms",
            "order_status": "/order/{order_id}",
            "analytics": "/analytics"
        }
    }


@app.post("/voice")
async def voice_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Main Twilio voice webhook handler.
    Handles incoming calls and initiates the ordering conversation.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", str(uuid.uuid4()))
    caller = form_data.get("From", "Unknown")
    
    logger.info(f"📞 Incoming call from {caller} - CallSid: {call_sid}")
    
    # Initialize or retrieve session
    session = session_manager.get_or_create(call_sid)
    session["caller"] = caller
    session["start_time"] = datetime.now().isoformat()
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Welcome message
    welcome_text = (
        "Welcome to Bella's Italian Kitchen! "
        "I'm your AI ordering assistant. "
        "I can help you browse our menu and place an order. "
        "What would you like today? You can ask about our pizzas, pastas, appetizers, or drinks."
    )
    
    # Use Gather to collect speech input
    gather = Gather(
        input="speech",
        action="/voice/process",
        method="POST",
        speechTimeout="auto",
        language="en-US",
        enhanced=True,
        speechModel="phone_call"
    )
    gather.say(welcome_text, voice="Polly.Joanna", language="en-US")
    response.append(gather)
    
    # Fallback if no input
    response.redirect("/voice/timeout")
    
    # Log the interaction
    background_tasks.add_task(
        log_interaction,
        call_sid=call_sid,
        direction="outgoing",
        text=welcome_text
    )
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/process")
async def process_voice_input(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Process speech input from the caller.
    Uses Whisper for transcription (if raw audio) or Twilio's transcription,
    then processes with LLM and responds.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "")
    speech_result = form_data.get("SpeechResult", "")
    confidence = form_data.get("Confidence", "0")
    
    logger.info(f"🎤 Speech input - CallSid: {call_sid}, Text: '{speech_result}', Confidence: {confidence}")
    
    # Get session context
    session = session_manager.get_or_create(call_sid)
    session.setdefault("conversation_history", [])
    session.setdefault("current_order", [])
    
    # Log user input
    background_tasks.add_task(
        log_interaction,
        call_sid=call_sid,
        direction="incoming",
        text=speech_result
    )
    
    # Add to conversation history
    session["conversation_history"].append({
        "role": "user",
        "content": speech_result,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process with LLM and RAG
    try:
        ai_response, order_update = await process_with_llm(speech_result, session)
    except Exception as e:
        logger.error(f"LLM processing error: {e}")
        ai_response = "I'm sorry, I had trouble understanding that. Could you please repeat?"
        order_update = None
    
    # Update order if needed
    if order_update:
        if order_update.get("action") == "add":
            session["current_order"].extend(order_update.get("items", []))
        elif order_update.get("action") == "remove":
            for item in order_update.get("items", []):
                if item in session["current_order"]:
                    session["current_order"].remove(item)
        elif order_update.get("action") == "clear":
            session["current_order"] = []
        elif order_update.get("action") == "confirm":
            # Process order confirmation
            order_id = await finalize_order(session, background_tasks)
            ai_response = f"Your order has been confirmed! Your order number is {order_id}. " \
                         f"Your total is ${session.get('order_total', 0):.2f}. " \
                         "Thank you for ordering with Bella's Italian Kitchen!"
    
    # Add AI response to history
    session["conversation_history"].append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Check for call completion keywords
    should_end_call = any(keyword in ai_response.lower() for keyword in [
        "thank you for ordering",
        "goodbye",
        "have a great day"
    ])
    
    # Create TwiML response
    response = VoiceResponse()
    
    if should_end_call:
        response.say(ai_response, voice="Polly.Joanna", language="en-US")
        response.hangup()
    else:
        gather = Gather(
            input="speech",
            action="/voice/process",
            method="POST",
            speechTimeout="auto",
            language="en-US",
            enhanced=True,
            speechModel="phone_call"
        )
        gather.say(ai_response, voice="Polly.Joanna", language="en-US")
        response.append(gather)
        response.redirect("/voice/timeout")
    
    # Log the response
    background_tasks.add_task(
        log_interaction,
        call_sid=call_sid,
        direction="outgoing",
        text=ai_response
    )
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/timeout")
async def voice_timeout(request: Request):
    """Handle timeout when user doesn't respond."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "")
    
    session = session_manager.get_or_create(call_sid)
    timeout_count = session.get("timeout_count", 0) + 1
    session["timeout_count"] = timeout_count
    
    response = VoiceResponse()
    
    if timeout_count >= 3:
        response.say(
            "I haven't heard from you in a while. "
            "If you'd like to place an order, please call back anytime. Goodbye!",
            voice="Polly.Joanna"
        )
        response.hangup()
    else:
        gather = Gather(
            input="speech",
            action="/voice/process",
            method="POST",
            speechTimeout="auto",
            language="en-US"
        )
        gather.say(
            "I'm still here! What would you like to order?",
            voice="Polly.Joanna"
        )
        response.append(gather)
        response.redirect("/voice/timeout")
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming SMS for order status or simple text orders."""
    form_data = await request.form()
    from_number = form_data.get("From", "")
    body = form_data.get("Body", "").strip().lower()
    
    logger.info(f"📱 SMS from {from_number}: {body}")
    
    response_text = "Thanks for texting Bella's Italian Kitchen! "
    
    if "status" in body:
        # Check order status
        orders = database.get_orders_by_phone(from_number)
        if orders:
            latest = orders[-1]
            response_text += f"Your latest order #{latest['id']} is {latest['status']}."
        else:
            response_text += "No recent orders found for this number."
    elif "menu" in body:
        response_text += "Visit our website or call us to hear our full menu! We have pizzas, pastas, appetizers, and more."
    else:
        response_text += "Reply MENU for our offerings, or STATUS to check your order. For full ordering, please call us!"
    
    # Create TwiML response for SMS
    from twilio.twiml.messaging_response import MessagingResponse
    twiml_response = MessagingResponse()
    twiml_response.message(response_text)
    
    return Response(content=str(twiml_response), media_type="application/xml")


@app.get("/order/{order_id}")
async def get_order_status(order_id: str):
    """Get order status by ID."""
    order = database.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.get("/analytics")
async def get_analytics():
    """Get basic analytics about calls and orders."""
    return {
        "total_calls": database.get_call_count(),
        "total_orders": database.get_order_count(),
        "popular_items": database.get_popular_items(),
        "average_order_value": database.get_average_order_value()
    }


@app.get("/menu")
async def get_menu():
    """Return the full menu as JSON."""
    return menu_rag.get_full_menu()


async def process_with_llm(user_input: str, session: dict) -> tuple[str, Optional[dict]]:
    """
    Process user input with LLM and RAG.
    Returns AI response and any order updates.
    """
    from utils.llm import LLMProcessor
    
    # Get relevant menu context via RAG
    menu_context = menu_rag.search(user_input, top_k=5)
    
    # Build conversation context
    conversation = session.get("conversation_history", [])[-10:]  # Last 10 turns
    current_order = session.get("current_order", [])
    
    # Calculate current total
    order_total = sum(item.get("price", 0) for item in current_order)
    
    # Create prompt with context
    prompt = get_order_prompt(
        user_input=user_input,
        menu_context=menu_context,
        conversation_history=conversation,
        current_order=current_order,
        order_total=order_total
    )
    
    # Process with LLM
    llm = LLMProcessor()
    response, order_update = await llm.process(prompt, user_input, current_order)
    
    # Update session total
    if order_update and order_update.get("items"):
        session["order_total"] = order_total + sum(
            item.get("price", 0) for item in order_update.get("items", [])
        )
    
    return response, order_update


async def finalize_order(session: dict, background_tasks: BackgroundTasks) -> str:
    """Finalize and save the order."""
    order_id = f"BIK-{uuid.uuid4().hex[:8].upper()}"
    
    order_data = {
        "id": order_id,
        "caller": session.get("caller", "Unknown"),
        "items": session.get("current_order", []),
        "total": session.get("order_total", 0),
        "status": "confirmed",
        "created_at": datetime.now().isoformat(),
        "conversation_history": session.get("conversation_history", [])
    }
    
    # Save to database
    database.save_order(order_data)
    
    # Simulate payment processing
    background_tasks.add_task(
        payment_processor.process_payment,
        order_id=order_id,
        amount=session.get("order_total", 0)
    )
    
    logger.info(f"✅ Order finalized: {order_id}")
    
    return order_id


def log_interaction(call_sid: str, direction: str, text: str):
    """Log call interactions for analytics."""
    log_entry = {
        "call_sid": call_sid,
        "direction": direction,
        "text": text,
        "timestamp": datetime.now().isoformat()
    }
    database.save_interaction(log_entry)
    logger.info(f"📝 Logged: [{direction}] {text[:50]}...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )

