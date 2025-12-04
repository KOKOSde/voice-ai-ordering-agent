"""
Prompt templates for the Voice-Order Restaurant AI Agent.
Contains system prompts, order processing prompts, and dynamic prompt builders.
"""

SYSTEM_PROMPT = """You are a friendly and helpful AI phone ordering assistant for Bella's Italian Kitchen, an authentic Italian restaurant. Your role is to help customers place orders over the phone.

## Your Personality
- Warm, friendly, and professional
- Patient with customers who need time to decide
- Knowledgeable about the menu and able to make recommendations
- Clear and concise in your responses (remember, this is a phone call)
- Natural conversational tone (avoid sounding robotic)

## Your Capabilities
1. Help customers browse the menu and answer questions about dishes
2. Take food orders and customize them (toppings, sizes, dietary restrictions)
3. Provide recommendations based on preferences
4. Calculate order totals
5. Confirm orders before finalizing
6. Handle special requests and dietary needs

## Important Guidelines
- Keep responses conversational and relatively brief (under 50 words when possible for phone calls)
- Always confirm items before adding to order
- Mention prices when discussing items
- Ask clarifying questions when orders are ambiguous
- Summarize the current order when asked or when it makes sense
- For pizzas, always ask about size if not specified (Personal $14.99, Medium +$4, Large +$8)
- Suggest popular items or complements when appropriate
- Be helpful with dietary restrictions (gluten-free, vegetarian, vegan options available)

## Response Format
Respond naturally as if speaking on the phone. Do not use bullet points, markdown, or special formatting - just speak naturally.

## Order Management
When a customer wants to order something:
1. Confirm the item and any customizations
2. Mention the price
3. Ask if they'd like anything else

When confirming final order:
1. Read back all items with prices
2. State the total
3. Ask for confirmation
4. Thank them for their order
"""

INTENT_RECOGNITION_PROMPT = """Analyze the customer's input and determine their intent. Output a JSON object.

Possible intents:
- "browse_menu": Customer wants to hear about menu items or categories
- "add_item": Customer wants to order something
- "remove_item": Customer wants to remove something from their order
- "modify_item": Customer wants to change something in their order
- "check_order": Customer wants to hear their current order
- "confirm_order": Customer is ready to confirm and pay
- "cancel_order": Customer wants to cancel everything
- "ask_question": Customer has a question about an item
- "get_recommendation": Customer wants a suggestion
- "other": General conversation or unclear intent

Also extract:
- items: List of menu items mentioned (if any)
- quantity: Number of items (default 1)
- customizations: Any modifications requested
- size: Pizza size if mentioned (personal/medium/large)

Customer said: "{user_input}"

Respond with JSON only:
"""

def get_order_prompt(
    user_input: str,
    menu_context: str,
    conversation_history: list,
    current_order: list,
    order_total: float
) -> str:
    """
    Build a complete prompt for the LLM with all context.
    
    Args:
        user_input: The customer's latest message
        menu_context: Relevant menu items from RAG search
        conversation_history: Recent conversation turns
        current_order: List of items in the current order
        order_total: Current order total
    
    Returns:
        Complete prompt string for the LLM
    """
    # Format conversation history
    history_text = ""
    for turn in conversation_history[-6:]:  # Last 6 turns for context
        role = "Customer" if turn["role"] == "user" else "You"
        history_text += f"{role}: {turn['content']}\n"
    
    # Format current order
    if current_order:
        order_text = "Current order:\n"
        for i, item in enumerate(current_order, 1):
            customizations = ""
            if item.get("customizations"):
                customizations = f" ({', '.join(item['customizations'])})"
            order_text += f"  {i}. {item['name']}{customizations} - ${item['price']:.2f}\n"
        order_text += f"  Subtotal: ${order_total:.2f}"
    else:
        order_text = "Current order: Empty"
    
    prompt = f"""{SYSTEM_PROMPT}

## Relevant Menu Information
{menu_context}

## Conversation So Far
{history_text}

## {order_text}

## Latest Customer Message
Customer: {user_input}

## Your Response
Respond naturally as a phone order assistant. Remember to be conversational and keep it brief for phone calls:"""
    
    return prompt


def get_menu_search_prompt(query: str) -> str:
    """Build a prompt for understanding what menu items to search for."""
    return f"""Given this customer query, extract the menu items or categories they're interested in.

Query: "{query}"

Extract:
- Categories: appetizers, pizzas, pastas, entrees, salads, desserts, drinks, wine
- Specific items mentioned
- Dietary preferences: vegetarian, vegan, gluten-free
- Keywords for search

Respond with comma-separated search terms:"""


def get_recommendation_prompt(
    preferences: str,
    dietary_restrictions: list,
    menu_items: list
) -> str:
    """Build a prompt for generating recommendations."""
    restrictions_text = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
    items_text = "\n".join([f"- {item['name']}: {item['description']} (${item['price']})" 
                           for item in menu_items])
    
    return f"""Based on the customer's preferences and our menu, suggest 2-3 items.

Customer preferences: {preferences}
Dietary restrictions: {restrictions_text}

Available items:
{items_text}

Give a brief, friendly recommendation suitable for a phone conversation:"""


def get_order_summary_prompt(order_items: list, total: float) -> str:
    """Build a prompt for summarizing the order."""
    items_text = "\n".join([
        f"- {item['name']}" + (f" ({item.get('size', 'regular')})" if item.get('size') else "") +
        (f" with {', '.join(item.get('customizations', []))}" if item.get('customizations') else "") +
        f" - ${item['price']:.2f}"
        for item in order_items
    ])
    
    return f"""Summarize this order naturally for a phone conversation:

Items:
{items_text}

Subtotal: ${total:.2f}
Tax (8.5%): ${total * 0.085:.2f}
Total: ${total * 1.085:.2f}

Give a brief, clear summary:"""


# Chain of thought prompts for complex reasoning
COT_ORDER_ANALYSIS = """Let me analyze this order step by step:

1. What did the customer explicitly ask for?
2. What details are missing that I need to ask about?
3. Does this match anything on our menu?
4. What's the correct price?
5. Should I suggest any add-ons or modifications?

Customer said: "{user_input}"

Thinking through this:"""


# Few-shot examples for intent recognition
INTENT_EXAMPLES = """
Example 1:
Customer: "What pizzas do you have?"
Intent: browse_menu
Category: pizzas

Example 2:
Customer: "I'll have a large pepperoni pizza"
Intent: add_item
Item: Pepperoni Pizza
Size: large
Quantity: 1

Example 3:
Customer: "Actually, remove the garlic bread"
Intent: remove_item
Item: Garlic Bread

Example 4:
Customer: "What's in the Quattro Formaggi?"
Intent: ask_question
Item: Quattro Formaggi

Example 5:
Customer: "What do you recommend for someone who doesn't eat meat?"
Intent: get_recommendation
Preference: vegetarian

Example 6:
Customer: "That's everything, I'm ready to order"
Intent: confirm_order

Now analyze:
Customer: "{user_input}"
"""


# Error handling prompts
CLARIFICATION_PROMPT = """I didn't quite catch that. The customer said: "{user_input}"

Possible interpretations:
{possibilities}

Generate a polite clarifying question:"""


FALLBACK_RESPONSES = [
    "I'm sorry, I didn't quite catch that. Could you repeat what you'd like to order?",
    "I want to make sure I get your order right. Could you say that again?",
    "Sorry about that - could you repeat your order?",
    "I missed that - what would you like to add to your order?",
]

