"""The same balance tool and question, with LangChain managing the loop."""

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "You are a reconciliation assistant. Obtain balances using the available tool. "
    "Never invent figures or a cause. After receiving the balances, report the "
    "totals and numerical difference concisely; explain that the cause is not "
    "yet established."
)
QUESTION = "What are the System A and System B balances for case C001, and what is the difference?"


# PART 1: LangChain builds the tool description from this function.
# The docstring describes its purpose; the type hint describes its input.
@tool
def get_case_balances(case_id: Literal["C001", "C002", "C003"]) -> dict:
    """Read local case data and calculate both system totals and A-minus-B in Python."""
    if case_id not in {"C001", "C002", "C003"}:
        return {"error": "Unknown case. Use C001, C002 or C003."}
    a = json.loads((ROOT / "data/system_a.json").read_text())
    b = json.loads((ROOT / "data/system_b.json").read_text())
    a_total = sum(row["amount_inr"] for row in a if row["case_id"] == case_id)
    b_total = sum(row["amount_inr"] for row in b if row["case_id"] == case_id)
    print(f"\nPYTHON EXECUTES: get_case_balances({case_id!r})", flush=True)
    return {
        "case_id": case_id,
        "currency": "INR",
        "system_a_total": a_total,
        "system_b_total": b_total,
        "difference_a_minus_b": a_total - b_total,
    }


def main():
    load_dotenv(ROOT / ".env", override=True)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Add OPENROUTER_API_KEY to the .env beside this script.")

    # PART 2: This adapter connects LangChain to OpenRouter.
    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        max_tokens=2048,
        max_retries=0,
        timeout=60000,  # This integration uses milliseconds: 60 seconds.
        disable_streaming=True,  # Show completed messages, not individual tokens.
    )

    # PART 3: create_agent builds the model -> tools -> model loop.
    agent = create_agent(
        model=model,
        tools=[get_case_balances],
        system_prompt=SYSTEM_PROMPT,
    )

    # PART 4: This loop OBSERVES progress. LangChain controls tool execution,
    # appends messages, and decides whether to make another model call.
    # No trimming/summarization/checkpointing has been configured in this lesson.
    executed_tools = 0
    for update in agent.stream(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config={"recursion_limit": 12},  # Cap graph steps; not a token/context limit.
        stream_mode="updates",
    ):
        for step, state in update.items():
            print(f"\nLANGCHAIN STEP: {step}", flush=True)
            for message in state.get("messages", []):
                if message.type == "ai":
                    metadata = message.response_metadata
                    print("Finish reason:", metadata.get("finish_reason", "not reported"))
                    print("Token usage:", message.usage_metadata)
                    if metadata.get("finish_reason") == "length":
                        raise SystemExit("Response reached its token limit. Send this output so we can adjust it.")
                    if message.tool_calls:
                        for call in message.tool_calls:
                            print("MODEL REQUESTED TOOL:", call["name"])
                            print("Arguments:", json.dumps(call["args"]))
                    else:
                        print("FINAL ANSWER:", message.content)
                elif message.type == "tool":
                    print("TOOL RESULT:", message.content)
                    print("Tool status:", message.status)
                    if message.status == "success":
                        executed_tools += 1

    print("\nSuccessful tool responses:", executed_tools)
    if executed_tools == 0:
        print("No successful tool response. This run has not demonstrated successful tool use.")


if __name__ == "__main__":
    main()
