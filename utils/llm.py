"""
LLM processing utilities for intent recognition and response generation.
Supports multiple backends: Hugging Face Transformers, OpenAI API, and local models.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model instances (lazy loaded)
_llm_model = None
_llm_tokenizer = None


class LLMProcessor:
    """
    Language Model processor for order intent recognition and response generation.
    Uses Hugging Face Transformers with Mistral or Qwen2-Audio models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize LLM processor.

        Args:
            model_name: Model to use (default from environment)
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # Determine backend
        if self.openai_key:
            self.backend = "openai"
        elif self.hf_token:
            self.backend = "huggingface"
        else:
            self.backend = "local"

        self._model = None
        self._tokenizer = None
        self._initialized = False

        logger.info(f"LLM Processor initialized with backend: {self.backend}")

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        if self.backend == "huggingface":
            self._init_huggingface()
        elif self.backend == "local":
            self._init_local()
        # OpenAI doesn't need initialization

        self._initialized = True

    def _init_huggingface(self):
        """Initialize Hugging Face model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading HuggingFace model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )

            logger.info("HuggingFace model loaded successfully")

        except ImportError as e:
            logger.warning(f"HuggingFace transformers not available: {e}")
            self.backend = "fallback"
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            self.backend = "fallback"

    def _init_local(self):
        """Initialize a smaller local model for testing."""
        try:
            from transformers import pipeline

            # Use a smaller model for local testing
            self._model = pipeline("text-generation", model="distilgpt2", max_length=150)

            logger.info("Local model (distilgpt2) loaded for testing")

        except ImportError:
            logger.warning("Transformers not available, using fallback")
            self.backend = "fallback"

    async def process(
        self, prompt: str, user_input: str, current_order: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process user input and generate response.

        Args:
            prompt: Full prompt with context
            user_input: User's latest message
            current_order: Current order items

        Returns:
            Tuple of (response_text, order_update)
        """
        self._ensure_initialized()

        # First, extract intent
        intent_data = self._extract_intent(user_input)

        # Generate response based on backend
        if self.backend == "openai":
            response = await self._generate_openai(prompt)
        elif self.backend == "huggingface":
            response = self._generate_huggingface(prompt)
        elif self.backend == "local":
            response = self._generate_local(prompt)
        else:
            response = self._generate_fallback(user_input, intent_data, current_order)

        # Determine order updates from intent
        order_update = self._process_intent(intent_data, user_input)

        return response, order_update

    def _extract_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Extract intent and entities from user input.
        Uses pattern matching as a fast first pass.
        """
        input_lower = user_input.lower()

        intent = {"type": "unknown", "items": [], "action": None, "quantity": 1}

        # Intent detection patterns
        if any(word in input_lower for word in ["menu", "what do you have", "options", "tell me about"]):
            intent["type"] = "browse_menu"
            # Extract category
            for category in [
                "pizza",
                "pasta",
                "appetizer",
                "dessert",
                "drink",
                "salad",
                "wine",
            ]:
                if category in input_lower:
                    intent["category"] = category + "s" if not category.endswith("s") else category

        elif any(
            word in input_lower
            for word in [
                "i'll have",
                "i want",
                "i'd like",
                "can i get",
                "order",
                "add",
                "give me",
            ]
        ):
            intent["type"] = "add_item"
            intent["action"] = "add"
            intent["items"] = self._extract_items(input_lower)

        elif any(word in input_lower for word in ["remove", "cancel", "take off", "no more", "delete"]):
            intent["type"] = "remove_item"
            intent["action"] = "remove"
            intent["items"] = self._extract_items(input_lower)

        elif any(
            word in input_lower
            for word in [
                "that's it",
                "that's all",
                "confirm",
                "done",
                "place order",
                "checkout",
                "pay",
            ]
        ):
            intent["type"] = "confirm_order"
            intent["action"] = "confirm"

        elif any(
            word in input_lower
            for word in [
                "what's in my order",
                "my order",
                "what did i order",
                "order so far",
            ]
        ):
            intent["type"] = "check_order"

        elif any(word in input_lower for word in ["recommend", "suggest", "what's good", "popular", "favorite"]):
            intent["type"] = "get_recommendation"

        elif "?" in user_input or any(word in input_lower for word in ["what is", "how much", "does it", "is there"]):
            intent["type"] = "ask_question"

        # Extract quantity
        quantity_match = re.search(r"\b(one|two|three|four|five|six|\d+)\b", input_lower)
        if quantity_match:
            quantity_map = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
            }
            q = quantity_match.group(1)
            intent["quantity"] = quantity_map.get(q, int(q) if q.isdigit() else 1)

        # Extract size for pizzas
        for size in ["personal", "medium", "large"]:
            if size in input_lower:
                intent["size"] = size
                break

        return intent

    def _extract_items(self, text: str) -> List[str]:
        """Extract menu item names from text."""
        # Common menu items to look for
        menu_items = [
            "margherita",
            "pepperoni",
            "quattro formaggi",
            "meat lovers",
            "vegetarian",
            "hawaiian",
            "bbq chicken",
            "truffle mushroom",
            "spaghetti",
            "fettuccine",
            "lasagna",
            "carbonara",
            "penne",
            "bruschetta",
            "calamari",
            "caprese",
            "garlic bread",
            "mozzarella sticks",
            "caesar salad",
            "house salad",
            "tiramisu",
            "cannoli",
            "gelato",
            "pizza",
            "pasta",
            "salad",
            "soup",
            "wine",
            "beer",
            "soda",
        ]

        found = []
        text_lower = text.lower()

        for item in menu_items:
            if item in text_lower:
                found.append(item.title())

        return found

    def _process_intent(self, intent: Dict[str, Any], user_input: str) -> Optional[Dict[str, Any]]:
        """Convert intent to order update."""
        if intent["action"] == "add" and intent["items"]:
            # Build order items with estimated prices
            from utils.rag import MenuRAG

            rag = MenuRAG()

            items = []
            for item_name in intent["items"]:
                menu_item = rag.get_item_by_name(item_name)
                if menu_item:
                    order_item = {
                        "name": menu_item["name"],
                        "price": menu_item["price"],
                        "quantity": intent.get("quantity", 1),
                        "size": intent.get("size"),
                        "customizations": [],
                    }
                    items.append(order_item)
                else:
                    # Fallback with estimated price
                    items.append(
                        {
                            "name": item_name,
                            "price": 15.99,  # Default price
                            "quantity": intent.get("quantity", 1),
                            "size": intent.get("size"),
                            "customizations": [],
                        }
                    )

            return {"action": "add", "items": items}

        elif intent["action"] == "remove" and intent["items"]:
            return {"action": "remove", "items": intent["items"]}

        elif intent["action"] == "confirm":
            return {"action": "confirm", "items": []}

        return None

    async def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.openai_key)

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly restaurant phone ordering assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback("", {}, [])

    def _generate_huggingface(self, prompt: str) -> str:
        """Generate response using HuggingFace model."""
        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part
            response = response[len(prompt) :].strip()

            # Clean up the response
            response = self._clean_response(response)

            return response

        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._generate_fallback("", {}, [])

    def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            result = self._model(prompt, max_length=150, num_return_sequences=1)
            return self._clean_response(result[0]["generated_text"])
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return self._generate_fallback("", {}, [])

    def _generate_fallback(
        self,
        user_input: str,
        intent: Dict[str, Any],
        current_order: List[Dict[str, Any]],
    ) -> str:
        """
        Fallback response generation using templates.
        Used when ML models are unavailable.
        """
        intent_type = intent.get("type", "unknown")

        responses = {
            "browse_menu": [
                "We have a great selection! Our popular items include our Margherita Pizza, Spaghetti Bolognese, and Tiramisu for dessert. What sounds good to you?",
                "Let me tell you about our menu. We have pizzas, pastas, appetizers, salads, and desserts. What are you in the mood for?",
            ],
            "add_item": [
                "Great choice! I've added that to your order. Would you like anything else?",
                "Perfect, I've got that down. Can I get you anything else?",
            ],
            "remove_item": [
                "No problem, I've removed that from your order. Anything else you'd like to change?",
            ],
            "check_order": self._format_order_response(current_order),
            "confirm_order": [
                "Excellent! Let me confirm your order and we'll get started on that right away.",
            ],
            "get_recommendation": [
                "I'd recommend our Margherita Pizza - it's a customer favorite! If you're in the mood for pasta, our Spaghetti Bolognese is amazing. And don't miss our Tiramisu for dessert!",
            ],
            "ask_question": [
                "Great question! Let me help you with that. What would you like to know about our menu?",
            ],
            "unknown": [
                "I'd be happy to help you with your order. What would you like today?",
            ],
        }

        import random

        response_list = responses.get(intent_type, responses["unknown"])
        if isinstance(response_list, str):
            return response_list
        return random.choice(response_list)

    def _format_order_response(self, current_order: List[Dict[str, Any]]) -> str:
        """Format current order as a spoken response."""
        if not current_order:
            return "Your order is currently empty. What would you like to add?"

        items = ", ".join([item.get("name", "item") for item in current_order])
        total = sum(item.get("price", 0) for item in current_order)

        return f"So far you have: {items}. Your current total is ${total:.2f}. Would you like to add anything else?"

    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove any markdown or special characters
        response = re.sub(r"[*#`]", "", response)

        # Remove any obvious prompt leakage
        for phrase in ["Customer:", "Assistant:", "AI:", "User:"]:
            if phrase in response:
                response = response.split(phrase)[0]

        # Truncate at natural ending
        for ending in [". ", "! ", "? "]:
            if ending in response:
                parts = response.rsplit(ending, 1)
                if len(parts) > 1 and len(parts[0]) > 20:
                    response = parts[0] + ending[0]

        return response.strip()

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of customer message."""
        text_lower = text.lower()

        positive_words = [
            "great",
            "thanks",
            "perfect",
            "love",
            "awesome",
            "delicious",
            "wonderful",
        ]
        negative_words = [
            "bad",
            "wrong",
            "upset",
            "angry",
            "terrible",
            "horrible",
            "awful",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": 0.7,
            "positive_words": positive_count,
            "negative_words": negative_count,
        }
