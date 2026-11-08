# src/refund_agent.py (Part D, tasks 5-8): a refund agent that calls tools in a ReAct loop.
# ReAct loop: ask the LLM -> it asks for a tool -> we run the tool -> we give the result back -> repeat.
# Run from the main folder:  python -m src.refund_agent

import json
import os
import sqlite3
import time
from datetime import datetime

import pandas as pd

import mock_llm
from src.policy_rag import PolicyRAG

DB_PATH = "data/instamart.db"
RUNS_FILE = "outputs/agent_runs.jsonl"
LEDGER_FILE = "outputs/refund_ledger.jsonl"
STATE_FOLDER = "state/"
SYSTEM_PROMPT = "AGENT: you handle refund requests. Use the tools to check the order and follow the refund policy."

rag = PolicyRAG()   # reuse the retrieval built in tasks 1-3


# ---------------- Task 5: tools ----------------

def get_order(order_id):
    connection = sqlite3.connect(DB_PATH)
    row = connection.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)).fetchone()
    connection.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": rag.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    # Completed orders reach this tool. More than 45 minutes -> full refund, otherwise 0.
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(order_amount, 2)}
    return {"refund_inr": 0}


def write_ledger(tool_name, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    line = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool_name,
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def issue_refund(order_id, refund_inr):
    write_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    write_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Work out the refund in INR from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue a refund of INR 1000 or less and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

# Tool name -> Python function
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------- Tasks 6-8: the agent loop ----------------

def save_checkpoint(run):
    os.makedirs(STATE_FOLDER, exist_ok=True)
    with open(STATE_FOLDER + run["run_id"] + ".json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)


def fallback(run, status, reason):
    tools_done = [step["tool"] for step in run["steps"]]
    summary = (status + ": " + reason + " Handing over to a human agent. Request: '" + run["user_message"]
               + "'. Tools completed so far: " + (", ".join(tools_done) if tools_done else "none") + ".")
    return finish(run, status, summary)


def finish(run, status, final_answer):
    return {"run_id": run["run_id"], "scenario_id": run["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": run["steps"], "llm_calls": run["llm_calls"],
            "tokens_used": run["tokens_used"]}


def agent_loop(run, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while True:
        # (c) time limit, checked at the top of every iteration
        if time.monotonic() - start >= time_limit_seconds:
            return fallback(run, "FALLBACK_TIMEOUT", "time limit reached.")
        # (a) iteration limit
        if run["llm_calls"] >= max_iterations:
            return fallback(run, "FALLBACK_MAX_ITERATIONS", "too many steps without a final answer.")

        response = mock_llm.chat(run["messages"], tools=TOOLS_SPEC)
        run["llm_calls"] += 1
        run["tokens_used"] += response.usage.total_tokens

        # (b) token limit, checked right after each model call (the requested tool is NOT run)
        if run["tokens_used"] > token_limit:
            return fallback(run, "FALLBACK_TOKEN_LIMIT", "token limit exceeded.")

        reply = response.choices[0].message
        if not reply.tool_calls:                         # no tool requested -> final answer
            status = reply.content.split(":")[0]
            return finish(run, status, reply.content)

        tool_call = reply.tool_calls[0]
        run["messages"].append({"role": "assistant", "content": None,
                                "tool_calls": [{"id": tool_call.id, "name": tool_call.name,
                                                "arguments": tool_call.arguments}]})
        observation = TOOL_MAP[tool_call.name](**tool_call.arguments)   # run the tool
        run["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name,
                                "content": json.dumps(observation)})
        run["steps"].append({"tool": tool_call.name, "arguments": tool_call.arguments,
                             "observation": observation})
        run["tools_executed"].append(tool_call.name)
        run["step_number"] = len(run["steps"])
        save_checkpoint(run)                             # task 8: checkpoint after every observation

        if crash_after_tools is not None and len(run["tools_executed"]) == crash_after_tools:
            raise RuntimeError("Simulated crash after tool number " + str(crash_after_tools))


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    run = {
        "run_id": run_id or scenario_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_message}],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
        "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                   "time_limit_seconds": time_limit_seconds},
    }
    return agent_loop(run, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id):
    with open(STATE_FOLDER + run_id + ".json", encoding="utf-8") as f:
        run = json.load(f)
    limits = run["limits"]
    return agent_loop(run, limits["max_iterations"], limits["token_limit"], limits["time_limit_seconds"])


def log_run(result):
    line = {key: result[key] for key in ["scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"]}
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def refund_amount(result):
    # The refund is the refund_inr of the issue_refund / escalate_to_human step, if any
    for step in result["steps"]:
        if step["tool"] in ["issue_refund", "escalate_to_human"]:
            return step["arguments"]["refund_inr"]
    return 0


def main():
    os.makedirs("outputs", exist_ok=True)
    for path in [RUNS_FILE, LEDGER_FILE]:       # start fresh so re-runs do not pile up
        if os.path.exists(path):
            os.remove(path)

    print("=== Task 6: the seven scenarios ===")
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for _, s in scenarios.iterrows():
        result = run_agent(s["message"], s["scenario_id"])
        log_run(result)
        rows.append({"scenario": s["scenario_id"], "expected": s["expected_outcome"],
                     "status": result["status"], "refund_inr": refund_amount(result),
                     "expected_refund_inr": s["expected_refund_inr"],
                     "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved", RUNS_FILE, "and", LEDGER_FILE)

    s1 = scenarios.iloc[0]["message"]
    print("\n=== Task 7: stop safely ===")
    demo = run_agent(s1, "S1", run_id="S1-token-demo", token_limit=200)
    print("S1 with token_limit=200 ->", demo["status"], "| llm_calls:", demo["llm_calls"],
          "| tokens:", demo["tokens_used"])
    demo = run_agent(s1, "S1", run_id="S1-timeout-demo", time_limit_seconds=0)
    print("S1 with time_limit_seconds=0 ->", demo["status"], "| llm_calls:", demo["llm_calls"])
    print("handoff summary:", demo["final_answer"])

    print("\n=== Task 8: crash and resume (S1) ===")
    run_id = "S1-crash-demo"
    try:
        run_agent(s1, "S1", run_id=run_id, crash_after_tools=2)
    except RuntimeError as error:
        print("crash:", error)
    with open(STATE_FOLDER + run_id + ".json", encoding="utf-8") as f:
        print("tools in the checkpoint before resuming:", json.load(f)["tools_executed"])
    resumed = resume_agent(run_id)
    print("tools executed across crash + resume:", [step["tool"] for step in resumed["steps"]])
    print("final:", resumed["status"], "| refund INR:", refund_amount(resumed), "|", resumed["final_answer"])


if __name__ == "__main__":
    main()
