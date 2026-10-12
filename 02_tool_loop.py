"""A visible, handwritten agent loop with one balance-calculation tool."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent


# PART 1: The tool is an ordinary Python function. It runs on your Mac.
def get_case_balances(case_id: str) -> dict:
    if case_id not in {"C001", "C002", "C003"}:
        return {"error": "Unknown case. Use C001, C002 or C003."}
    a = json.loads((ROOT / "data/system_a.json").read_text())
    b = json.loads((ROOT / "data/system_b.json").read_text())
    a_rows = [row for row in a if row["case_id"] == case_id]
    b_rows = [row for row in b if row["case_id"] == case_id]
    a_total = sum(row["amount_inr"] for row in a_rows)
    b_total = sum(row["amount_inr"] for row in b_rows)
    return {
        "case_id": case_id,
        "currency": "INR",
        "system_a_total": a_total,
        "system_b_total": b_total,
        "difference_a_minus_b": a_total - b_total,
    }


# PART 2: The model receives this description, not the Python source above.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_case_balances",
        "description": "Read the local case data and calculate both system totals and A-minus-B in Python.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "enum": ["C001", "C002", "C003"]}
            },
            "required": ["case_id"],
            "additionalProperties": False,
        },
    },
}]


def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Add OPENROUTER_API_KEY to the .env beside this script.")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=60.0,
        max_retries=0,
    )
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    # PART 3: This list is the conversation history stored by our application.
    messages = [
        {"role": "system", "content": "You are a reconciliation assistant. Obtain balances using the available tool. Never invent figures or a cause. After receiving the balances, report the totals and numerical difference concisely; explain that the cause is not yet established."},
        {"role": "user", "content": "What are the System A and System B balances for case C001, and what is the difference?"},
    ]
    executed_tools = 0

    # PART 4: Call the model, execute requested tools, add results, then repeat.
    for turn in range(1, 7):
        print(f"\nMODEL CALL {turn}: sending {len(messages)} messages plus the tool description")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=2048,
        )
        message = response.choices[0].message
        print("Model served:", response.model)
        print("Finish reason:", response.choices[0].finish_reason)

        if response.choices[0].finish_reason == "length":
            raise SystemExit("The response reached its token limit. Send this output so we can adjust the request.")

        if not message.tool_calls:
            print("\nFINAL ANSWER:", message.content)
            print("Tools executed:", executed_tools)
            if executed_tools == 0:
                print("No tool was requested. This run has not demonstrated tool use.")
            return

        # Preserve the assistant's tool request (including provider metadata).
        messages.append(message.model_dump(exclude_none=True))

        for call in message.tool_calls:
            name = call.function.name
            print("\nMODEL REQUESTED TOOL:", name)
            print("Arguments:", call.function.arguments)
            print("Call ID:", call.id)

            # Dispatch only the function we explicitly exposed.
            try:
                args = json.loads(call.function.arguments)
                if name != "get_case_balances":
                    result = {"error": "Unknown tool."}
                elif not isinstance(args, dict) or set(args) != {"case_id"}:
                    result = {"error": "Provide only case_id."}
                elif not isinstance(args["case_id"], str):
                    result = {"error": "case_id must be a string."}
                else:
                    print("PYTHON EXECUTES:", f"get_case_balances({args['case_id']!r})")
                    result = get_case_balances(**args)
                    executed_tools += 1
            except json.JSONDecodeError:
                result = {"error": "Arguments must be valid JSON."}

            print("TOOL RESULT:", json.dumps(result, indent=2))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
            print("Tool result added to history for the next model call.")

    raise SystemExit("Stopped after six model calls without a final answer.")


if __name__ == "__main__":
    main()
