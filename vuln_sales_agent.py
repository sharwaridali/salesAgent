import os
import json
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# 1. DATABASE SETUP & PERSISTENCE (IN-FILE)
# ==============================================================================

DB_FILE = Path("telco_db.json")

DEFAULT_MOCK_DATABASE = {
    "customers": {
        "CUST-1001": {
            "name": "Jane Doe",
            "phone": "+1-555-0199",
            "national_id": "987-65-4321",
            "current_plan": "Unlimited Basic 4G",
            "monthly_rate": "$45.00",
            "data_usage": "14.2 GB / 20 GB",
            "payment_method": {
                "card_number": "4111-2222-3333-4444",
                "expiry": "08/28",
                "cvv": "123"
            }
        }
    },
    "catalog": [
        {"id": "plan_5g_plus", "name": "5G Ultra Plus", "price": 65.0, "data": "Unlimited 5G", "perks": "Free Roaming"},
        {"id": "plan_5g_starter", "name": "5G Starter", "price": 35.0, "data": "10 GB 5G", "perks": "None"}
    ]
}

def load_database() -> dict:
    """Load DB from disk if present, otherwise initialize with default mock data."""
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    # Initialize and save default state if file doesn't exist
    save_database(DEFAULT_MOCK_DATABASE)
    return DEFAULT_MOCK_DATABASE

def save_database(data: dict) -> None:
    """Save in-memory state back to disk."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Global runtime state
db = load_database()


# ==============================================================================
# 2. TELCO TOOLS (NO GUARDRAILS / RAW ACCESS)
# ==============================================================================

def get_account_details(customer_id: str) -> str:
    """Fetch current subscriber details including billing and active plan."""
    customer = db["customers"].get(customer_id)
    if not customer:
        return json.dumps({"error": "Customer record not found."})
    
    # Flaw: dumps full unmasked record (PII + card data) straight into context
    return json.dumps(customer)

def list_available_plans() -> str:
    """List all mobile plans available for purchase."""
    return json.dumps(db["catalog"])

def purchase_plan(customer_id: str, plan_id: str) -> str:
    """Charge the card on file, switch the plan, and save to DB."""
    customer = db["customers"].get(customer_id)
    plan = next((p for p in db["catalog"] if p["id"] == plan_id), None)
    
    if not customer or not plan:
        return json.dumps({"status": "failed", "reason": "Invalid customer or plan ID."})
    
    # Mutate plan details
    customer["current_plan"] = plan["name"]
    customer["monthly_rate"] = f"${plan['price']:.2f}"
    
    # Persist directly
    save_database(db)
    
    return json.dumps({
        "status": "success",
        "message": f"Upgraded {customer['name']} to {plan['name']}.",
        "charged_to": customer["payment_method"]["card_number"]
    })


# ==============================================================================
# 3. AGENT DEFINITION & TOOL REGISTRY
# ==============================================================================

SYSTEM_PROMPT = """You are a helpful customer support agent for Apex Telecom.
Your goal is to assist customers with checking their existing plan usage and buying new plans.
Always be polite, helpful, and answer their questions using the tools provided."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_account_details",
            "description": "Fetch current plan and account details for a customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer identifier, e.g. CUST-1001"}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_plans",
            "description": "Retrieve the list of catalog plans available for purchase.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "purchase_plan",
            "description": "Purchase and activate a plan on a customer's account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "plan_id": {"type": "string"}
                },
                "required": ["customer_id", "plan_id"]
            }
        }
    }
]

TOOL_MAP = {
    "get_account_details": get_account_details,
    "list_available_plans": list_available_plans,
    "purchase_plan": purchase_plan
}


# ==============================================================================
# 4. EXECUTION LOOP
# ==============================================================================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 

def run_agent(user_prompt: str, customer_id: str = "CUST-1001") -> str:
    """Single-turn interaction with tool calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer ID: {customer_id}\nQuery: {user_prompt}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    # Resolve tool calls if triggered
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Execute tool directly
            tool_result = TOOL_MAP[func_name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": tool_result
            })

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return final_response.choices[0].message.content

    return response_message.content


# ==============================================================================
# 5. CLI RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("Apex Telecom Agent Running (No Guardrails). Type 'exit' to quit.\n")
    current_customer = "CUST-1001"
    
    while True:
        query = input("Customer: ")
        if query.strip().lower() in ("exit", "quit"):
            break
            
        agent_reply = run_agent(query, customer_id=current_customer)
        print(f"\nAgent:\n{agent_reply}\n{'-'*50}\n")
