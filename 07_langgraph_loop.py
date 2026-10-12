"""Build the familiar model/tool loop explicitly with LangGraph.

python3 07_langgraph_loop.py --preview
python3 07_langgraph_loop.py

Keep lessons 04 and 06 beside this file. Reuses their tools and prompts.
Preview builds the graph and prints initial input, but does not execute it.
"""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Imports define functions and tools; the imported scripts' main() guards do not run.
lesson06 = import_module("06_agent_rag")
ROOT = Path(__file__).resolve().parent
TOOLS = lesson06.TOOLS
SYSTEM_PROMPT = lesson06.SYSTEM_PROMPT
DEFAULT_QUESTION = lesson06.DEFAULT_QUESTION


def build_graph(model):
    # PART 1: This adapter will include tool descriptions in each model request.
    # bind_tools configures a callable model wrapper; it does NOT call the API.
    model_with_tools = model.bind_tools(TOOLS)

    # PART 2: A NODE is a function the graph can run. Its input is current state.
    def call_model(state: MessagesState):
        print("\n========== MODEL NODE ==========", flush=True)
        print("Messages currently in graph state:", len(state["messages"]))
        system = SystemMessage(content=SYSTEM_PROMPT)
        lesson06.print_model_input(system, state["messages"], TOOLS)

        # This line makes the model request: instructions + current message history.
        response = model_with_tools.invoke([system, *state["messages"]])
        print("Model served:", response.response_metadata.get("model_name", "not reported"))
        print("Finish reason:", response.response_metadata.get("finish_reason", "not reported"))
        print("Token usage:", response.usage_metadata)
        if response.response_metadata.get("finish_reason") == "length":
            raise SystemExit("Model reached its output limit. Share the output before another run.")
        if response.invalid_tool_calls:
            raise SystemExit("Model returned malformed tool arguments. Share the output before another run.")
        if response.tool_calls:
            for call in response.tool_calls:
                print("MODEL REQUESTED TOOL:", call["name"], json.dumps(call["args"]))
        else:
            print("FINAL ANSWER:", response.content)

        # Return an UPDATE, not the entire history.
        # MessagesState uses an add_messages reducer: new message IDs are appended;
        # updates with an existing ID replace that message. Our new reply is appended.
        return {"messages": [response]}

    # PART 3: This ordinary Python function chooses the next graph destination.
    # It runs AFTER the model reply has been merged into state.
    def choose_next_step(state: MessagesState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            print("ROUTING: tool requests present; next node is tools.", flush=True)
            return "tools"
        print("ROUTING: no tool requests; graph ends.", flush=True)
        return END

    # PART 4: Register the executable steps and the connections between them.
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    # ToolNode dispatches requested tool names, validates arguments, invokes Python,
    # and returns ToolMessages with matching call IDs. State merges those results.
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", choose_next_step, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")

    # compile creates the runnable graph. It does NOT execute any node yet.
    return builder.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Use the existing OPENROUTER_API_KEY in the .env beside this file.")
    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        max_tokens=2048, max_retries=0, timeout=60000, disable_streaming=True,
    )
    graph = build_graph(model)

    # PART 5: This is the initial state. The question enters the graph HERE.
    initial_state = {"messages": [{"role": "user", "content": args.question}]}
    print("QUESTION:", args.question, flush=True)
    if args.preview:
        from langchain_core.messages import HumanMessage
        lesson06.print_model_input(SystemMessage(content=SYSTEM_PROMPT), [HumanMessage(content=args.question)], TOOLS)
        print("\nRegistered nodes:", ", ".join(graph.nodes))
        print("Start runs model. Model routes to tools or end. Tools returns to model.")
        print("Preview complete: graph built, no nodes executed, no model or retrieval calls.")
        return

    # PART 6: Iterating stream starts execution. This for-loop observes updates;
    # the edges and choose_next_step function control the workflow.
    # No checkpointer or history trimming is configured. State lasts for this run.
    successful_tools = set()
    final_answer_seen = False
    for update in graph.stream(initial_state, config={"recursion_limit": 16}, stream_mode="updates"):
        for node_name, node_update in update.items():
            print("\nCOMPLETED NODE:", node_name, flush=True)
            for message in node_update.get("messages", []):
                print("NEW STATE MESSAGE:", message.type)
                if message.type == "tool":
                    print("TOOL RESULT:", message.name, "|", message.content)
                    print("Tool status:", message.status)
                    if message.status == "success":
                        successful_tools.add(message.name)
                elif message.type == "ai" and not message.tool_calls:
                    final_answer_seen = bool(message.content)
    print("\nSuccessful tools:", ", ".join(sorted(successful_tools)) or "none")
    print("Both lesson tools used successfully:", {item.name for item in TOOLS} <= successful_tools)
    print("Non-empty final answer received:", final_answer_seen)
    print("Graph run ended. Answer correctness and citation support still need review.")


if __name__ == "__main__":
    main()
