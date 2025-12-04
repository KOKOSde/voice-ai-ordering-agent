# 🍕 Voice AI Ordering Agent

> **A fully functional conversational AI system for restaurant phone ordering**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates an end-to-end voice AI ordering system for restaurants. Customers can call a phone number, have a natural conversation with an AI agent to browse the menu, place orders, and simulate payment — all through voice.

## 🎯 Features

- **📞 Twilio Voice Integration** - Handle incoming phone calls with Twilio Programmable Voice
- **🎤 Speech-to-Text** - Transcribe customer speech using OpenAI Whisper
- **🧠 LLM Processing** - Natural language understanding with Mistral/Qwen2-Audio via Hugging Face
- **🔍 RAG Menu Search** - Semantic search over menu items using FAISS vector database
- **🗣️ Text-to-Speech** - Generate voice responses with gTTS or ElevenLabs
- **💳 Simulated POS/Payments** - Mock Stripe integration and order management
- **🔄 Multi-turn Conversations** - Session state management with Redis/in-memory storage
- **📊 Analytics & Logging** - Call transcripts, order analytics, and error tracking

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Phone Call    │────▶│   Twilio Voice   │────▶│  FastAPI Server │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        │                                 ▼                                 │
                        │  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐  │
                        │  │   Whisper    │    │   LLM (Mistral)  │    │   FAISS RAG    │  │
                        │  │   (STT)      │───▶│   Intent + NLU   │◀──▶│   Menu Search  │  │
                        │  └──────────────┘    └──────────────────┘    └────────────────┘  │
                        │                                 │                                 │
                        │                                 ▼                                 │
                        │  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐  │
                        │  │  gTTS/11Labs │    │  Session Manager │    │   SQLite DB    │  │
                        │  │   (TTS)      │    │   (Redis/Mem)    │    │   Orders/Logs  │  │
                        │  └──────────────┘    └──────────────────┘    └────────────────┘  │
                        │                                                                   │
                        └───────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Twilio Account](https://www.twilio.com/try-twilio) (free trial available)
- [ngrok](https://ngrok.com/) for local development
- [Hugging Face Account](https://huggingface.co/) with API token

### 1. Clone the Repository

```bash
git clone https://github.com/KOKOSde/voice-ai-ordering-agent.git
cd voice-ai-ordering-agent
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

**Required variables:**
```bash
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
HUGGINGFACE_TOKEN=your_hf_token
```

### 4. Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Expose with ngrok

In a new terminal:
```bash
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

### 6. Configure Twilio Webhook

1. Go to [Twilio Console](https://console.twilio.com/)
2. Navigate to **Phone Numbers** → **Manage** → **Active Numbers**
3. Click on your phone number
4. Under **Voice & Fax**, set:
   - **A CALL COMES IN**: Webhook → `https://your-ngrok-url.ngrok.io/voice`
   - **HTTP POST**
5. Save

### 7. Make a Test Call! 📞

Call your Twilio phone number and start ordering!

## 📁 Project Structure

```
voice-ai-ordering-agent/
├── main.py                 # FastAPI application & Twilio webhooks
├── menu.json              # Restaurant menu database
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore patterns
├── Procfile              # Heroku deployment
├── utils/
│   ├── __init__.py
│   ├── transcription.py  # Whisper speech-to-text
│   ├── rag.py            # FAISS vector search for menu
│   ├── tts.py            # Text-to-speech (gTTS/ElevenLabs)
│   ├── llm.py            # LLM processing (Mistral/OpenAI)
│   ├── session.py        # Conversation state management
│   ├── database.py       # SQLite order storage
│   └── payment.py        # Simulated payment processing
├── tests/
│   └── test_main.py      # Unit tests
└── .github/
    └── workflows/
        └── ci.yml        # GitHub Actions CI
```

## 🎮 Demo Script

Here's a sample conversation flow:

```
📞 *Phone rings*

🤖 AI: "Welcome to Bella's Italian Kitchen! I'm your AI ordering assistant. 
        I can help you browse our menu and place an order. 
        What would you like today?"

👤 Customer: "What pizzas do you have?"

🤖 AI: "We have some great pizzas! Our most popular is the Margherita at $14.99, 
        the classic Pepperoni at $16.99, and our chef's special Truffle Mushroom 
        at $22.99. Would you like to hear more about any of these?"

👤 Customer: "I'll have a large pepperoni pizza"

🤖 AI: "Great choice! I've added a large Pepperoni Pizza for $24.99 to your order. 
        Would you like anything else? Maybe some garlic bread or a drink?"

👤 Customer: "Add garlic bread and that's it"

🤖 AI: "Perfect! I've added Garlic Bread for $5.99. Let me confirm your order:
        - Large Pepperoni Pizza: $24.99
        - Garlic Bread: $5.99
        Your total is $30.98 plus tax. Is that correct?"

👤 Customer: "Yes, that's correct"

🤖 AI: "Your order has been confirmed! Your order number is BIK-A7B3C2D1. 
        Your total is $33.51 including tax. Thank you for ordering with 
        Bella's Italian Kitchen! Have a great day!"
```

## 🔧 Configuration Options

### LLM Models

| Model | Description | Memory Required |
|-------|-------------|-----------------|
| `mistralai/Mistral-7B-Instruct-v0.2` | Default, good balance | ~16GB |
| `Qwen/Qwen2-Audio-7B-Instruct` | Audio-native model | ~16GB |
| `distilgpt2` | Local testing only | ~500MB |
| OpenAI API | Use `OPENAI_API_KEY` | Cloud-based |

### TTS Options

| Engine | Quality | Cost |
|--------|---------|------|
| gTTS | Good | Free |
| ElevenLabs | Excellent | Paid API |
| pyttsx3 | Basic (offline) | Free |

### Session Storage

| Backend | Use Case |
|---------|----------|
| In-memory | Development, single instance |
| Redis | Production, distributed |

## 🚢 Deployment

### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set TWILIO_ACCOUNT_SID=xxx
heroku config:set TWILIO_AUTH_TOKEN=xxx
heroku config:set HUGGINGFACE_TOKEN=xxx

# Deploy
git push heroku main
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway / Render

Both platforms auto-detect Python projects. Just connect your GitHub repo!

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/voice` | POST | Twilio voice webhook |
| `/voice/process` | POST | Process speech input |
| `/sms` | POST | SMS webhook |
| `/menu` | GET | Get full menu JSON |
| `/order/{id}` | GET | Get order status |
| `/analytics` | GET | Call/order analytics |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py -v
```

## 📈 Analytics Dashboard

The system logs all interactions for analytics:

- Call duration and outcomes
- Popular menu items
- Order values and trends
- Customer satisfaction signals

Access via `/analytics` endpoint.

## 🛡️ Security Considerations

- Never commit `.env` files
- Use Twilio request validation in production
- Rate limit API endpoints
- Sanitize all user inputs
- Use HTTPS in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Twilio](https://www.twilio.com/) for voice/SMS APIs
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [Hugging Face](https://huggingface.co/) for transformer models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

<p align="center">
  Made with ❤️ for the AI community
</p>

