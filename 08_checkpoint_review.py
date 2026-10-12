"""Save the model/tool loop in SQLite, pause for review, and resume later.

python3 08_checkpoint_review.py start
python3 08_checkpoint_review.py status
python3 08_checkpoint_review.py resume --decision approve

Keep lessons 04 and 06 beside this file. Start uses the existing .env and
retrieval index. Status and resume make no model or retrieval requests.
Progress is stored in checkpoints/lesson08.sqlite beside this script.
"""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

lesson06 = import_module("06_agent_rag")
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "checkpoints" / "lesson08.sqlite"
TOOLS = lesson06.TOOLS


# The messages field and its merge rule come from MessagesState.
# We add one ordinary field for the human's decision.
class ReviewState(MessagesState):
    review_decision: str


def make_model():
    load_dotenv(ROOT / ".env", override=True)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Use the existing OPENROUTER_API_KEY in the .env beside this file.")
    return ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        max_tokens=2048, max_retries=0, timeout=60000, disable_streaming=True,
    )


def build_graph(model, checkpointer):
    # Only start supplies a model. A resume at review must not call one again.
    model_with_tools = model.bind_tools(TOOLS) if model is not None else None

    def call_model(state: ReviewState):
        if model_with_tools is None:
            raise RuntimeError("This command must not execute a model node.")
        print("\n========== MODEL NODE ==========", flush=True)
        system = SystemMessage(content=lesson06.SYSTEM_PROMPT)
        lesson06.print_model_input(system, state["messages"], TOOLS)
        response = model_with_tools.invoke([system, *state["messages"]])
        print("Model served:", response.response_metadata.get("model_name", "not reported"))
        print("Finish reason:", response.response_metadata.get("finish_reason", "not reported"))
        print("Token usage:", response.usage_metadata)
        if response.response_metadata.get("finish_reason") == "length":
            raise SystemExit("Output limit reached. Share the output before another run.")
        if response.invalid_tool_calls:
            raise SystemExit("Malformed tool arguments. Share the output before another run.")
        if response.tool_calls:
            for call in response.tool_calls:
                print("MODEL REQUESTED TOOL:", call["name"], json.dumps(call["args"]))
        elif response.content:
            print("DRAFT FOR REVIEW:", response.content)
        else:
            raise SystemExit("The model returned no draft or tool request.")
        return {"messages": [response]}

    def choose_next_step(state: ReviewState):
        if state["messages"][-1].tool_calls:
            print("ROUTING: model -> tools", flush=True)
            return "tools"
        print("ROUTING: model -> review", flush=True)
        return "review"

    def review_draft(state: ReviewState):
        # This node re-enters from its beginning when resumed. Keep model calls
        # in the preceding node so resuming this node does not repeat them.
        print("\nENTER REVIEW NODE", flush=True)
        decision = interrupt({
            "question": "Accept this draft for the lesson? Choose approve or reject.",
            "draft": state["messages"][-1].content,
            "note": "Review acceptance does not establish a cause or reconcile the case.",
        })
        # First execution pauses above. On resume, interrupt returns the value
        # supplied in Command(resume=...), and execution reaches these lines.
        if decision not in ("approve", "reject"):
            raise ValueError("Review decision must be approve or reject.")
        print("REVIEW RECORDED:", decision, flush=True)
        return {"review_decision": decision}

    builder = StateGraph(ReviewState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("review", review_draft)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", choose_next_step, {"tools": "tools", "review": "review"}
    )
    builder.add_edge("tools", "model")
    builder.add_edge("review", END)
    # NEW: checkpoint writes go through this SQLite-backed component.
    return builder.compile(checkpointer=checkpointer)


def waiting_for_review(snapshot):
    return any(task.interrupts for task in snapshot.tasks)


def show_saved_state(graph, config):
    # A read from the checkpoint database; this does not run any graph node.
    snapshot = graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    print("\n========== SAVED STATE ==========", flush=True)
    print("Messages saved:", len(messages))
    print("Stored model replies:", sum(message.type == "ai" for message in messages))
    print("Next node(s):", ", ".join(snapshot.next) or "none")
    print("Review decision:", snapshot.values.get("review_decision", "not supplied"))
    if waiting_for_review(snapshot):
        print("STATUS: WAITING FOR REVIEW")
    elif snapshot.values.get("review_decision") in ("approve", "reject"):
        print("STATUS: REVIEW RECORDED; WORKFLOW ENDED")
    elif not snapshot.values:
        print("STATUS: NO SAVED RUN")
    else:
        print("STATUS: INCOMPLETE RUN; share the output before continuing")
    for message in reversed(messages):
        if message.type == "ai" and not message.tool_calls:
            print("SAVED DRAFT:", message.content)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "resume"))
    parser.add_argument("--thread", default="C001-review-1")
    parser.add_argument("--decision", choices=("approve", "reject"))
    args = parser.parse_args()
    if args.command == "resume" and args.decision is None:
        parser.error("resume needs --decision approve or --decision reject")
    if args.command != "resume" and args.decision is not None:
        parser.error("--decision is only used with resume")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("PROCESS ID:", os.getpid(), flush=True)
    print("CHECKPOINT FILE:", DB_PATH, flush=True)
    print("THREAD ID:", args.thread, flush=True)
    # thread_id identifies the saved execution, not a model's internal memory.
    config = {"configurable": {"thread_id": args.thread}, "recursion_limit": 16}

    # This opens a local database connection and closes it when the block ends.
    # The SQLite file remains after this Python process exits.
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        # Build the same workflow on each invocation, then load this thread.
        # Status and resume do not even construct an API model connection.
        graph = build_graph(None, checkpointer)
        snapshot = graph.get_state(config)
        if args.command == "status":
            show_saved_state(graph, config)
            print("\nStatus only: no nodes, model requests, or retrieval calls executed.")
            return
        if args.command == "start":
            if snapshot.values:
                raise SystemExit(
                    "This thread already has saved progress. Use status or resume. "
                    "For a new run, use start --thread C001-review-2."
                )
            graph = build_graph(make_model(), checkpointer)
            graph_input = {
                "messages": [{"role": "user", "content": lesson06.DEFAULT_QUESTION}],
                "review_decision": "pending",
            }
        else:
            if not waiting_for_review(snapshot):
                show_saved_state(graph, config)
                raise SystemExit("This thread is not paused for review. Nothing was resumed.")
            # Resume uses the SAME thread and the saved messages; it does not
            # submit the original question again or restart the model loop.
            graph_input = Command(resume=args.decision)

        for update in graph.stream(graph_input, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                print("\nPAUSED: progress saved; the Python command can finish.", flush=True)
                continue
            for node_name, node_update in update.items():
                print("\nCOMPLETED NODE:", node_name, flush=True)
                for message in node_update.get("messages", []):
                    if message.type == "tool":
                        print("TOOL RESULT:", message.name, "|", message.content)
                        print("Tool status:", message.status)
        show_saved_state(graph, config)
    print("\nCommand finished. The next command runs in a new Python process.")


if __name__ == "__main__":
    main()
