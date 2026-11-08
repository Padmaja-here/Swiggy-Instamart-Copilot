# src/triage.py (Part C): support-ticket triage with guardrails, using mock_llm.py.
# Run from the main folder:  python -m src.triage

import json
import os
import random
import re
import time

import pandas as pd

import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05          # seconds
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

# Task 2: redaction patterns, applied in this order
PATTERNS = [
    ("email", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    ("phone", r"\b[6-9]\d{9}\b", "[PHONE]"),
    ("order_id", r"ORD-\d{6}", "[ORDER_ID]"),
]

# Task 3: two system prompts
ZERO_SHOT = (
    "CLASSIFY the customer support ticket into exactly one of these categories: "
    "billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.'
)
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorized transaction"}
Ticket: My groceries arrived two hours late. -> {"category": "delivery", "reason": "late delivery"}
Ticket: The app crashes when I open my cart. -> {"category": "technical", "reason": "app crash"}"""


def redact(text):
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        counts[name] = n
    return text, counts


retry_count = 0   # how many retries happened in the current run


def call_with_retry(messages, ticket_id, show_log=True):
    # Retry up to 3 times on a temporary error, waiting a little longer each time
    global retry_count
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry_number * random.uniform(0.9, 1.15)
            retry_count += 1
            if show_log:
                print(f"  retry: ticket={ticket_id} retry_number={retry_number} wait={wait:.4f}s")
            time.sleep(wait)


def read_category(reply):
    # Returns (category, status)
    try:
        data = json.loads(reply)
        category = data.get("category") if isinstance(data, dict) else None
        if category in ALLOWED:
            return category, "classified_json"
        return "", "flagged_invalid_output"            # valid JSON but category not allowed
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", reply.lower())   # "billing-related" -> "billing", "related"
        found = set(word for word in words if word in ALLOWED)
        if len(found) == 1:
            return found.pop(), "classified_keyword"
        return "", "flagged_invalid_output"


def run_pipeline(prompt, token_budget=None, show_retries=True):
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    segments_path = "outputs/customer_segments.csv"
    segments = pd.read_csv(segments_path) if os.path.exists(segments_path) else pd.DataFrame(
        columns=["customer_id", "segment_name", "is_premium"])
    segment_name = dict(zip(segments["customer_id"], segments["segment_name"]))
    is_premium = dict(zip(segments["customer_id"], segments["is_premium"]))

    global retry_count
    mock_llm.reset()
    random.seed(42)
    retry_count = 0
    total_tokens = 0
    redaction_totals = {"email": 0, "phone": 0, "order_id": 0}
    rows = []

    for _, ticket in tickets.iterrows():
        text = ticket["text"]
        status, category, tokens = "", "", 0

        if text.strip() == "":
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            safe_text, counts = redact(text)
            for name in counts:
                redaction_totals[name] += counts[name]
            # Safety check: nothing private may be left in the prompt
            for _, pattern, _ in PATTERNS:
                assert re.search(pattern, safe_text) is None

            if token_budget is not None and total_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                messages = [{"role": "system", "content": prompt},
                            {"role": "user", "content": safe_text}]
                response = call_with_retry(messages, ticket["ticket_id"], show_retries)
                tokens = response.usage.total_tokens
                total_tokens += tokens
                category, status = read_category(response.choices[0].message.content or "")

        # Routing (task 7b)
        customer = ticket["customer_id"]
        premium = bool(is_premium.get(customer, False))
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        rows.append({"ticket_id": ticket["ticket_id"], "customer_id": customer, "status": status,
                     "category": category, "segment_name": segment_name.get(customer, ""),
                     "priority": priority, "route": route, "tokens_used": tokens,
                     "true_category": ticket["true_category"]})

    results = pd.DataFrame(rows)
    summary = {"total_tokens": total_tokens, "errors_recovered": retry_count,
               "redactions": redaction_totals}
    return results, summary


def report(name, results, summary):
    classified = results[results["status"].isin(["classified_json", "classified_keyword"])]
    correct = (classified["category"] == classified["true_category"]).sum()
    print(f"\n--- {name} ---")
    print("status counts:", results["status"].value_counts().to_dict())
    print("classified tickets:", len(classified))
    print(f"accuracy: {correct} of {len(classified)} = {correct / len(classified) * 100:.1f}%")
    print("total tokens:", summary["total_tokens"],
          "| tokens per classified ticket:", round(summary["total_tokens"] / len(classified), 1))
    print("transient errors recovered by a retry:", summary["errors_recovered"])
    return classified


def main():
    os.makedirs("outputs", exist_ok=True)
    columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]

    print("=== Tasks 1-2: guardrails and redaction ===")
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    empty = tickets["text"].str.strip() == ""
    long_tokens = tickets["text"].apply(mock_llm.est_tokens)
    print("empty tickets:", empty.sum(), "| long tickets:", (long_tokens > MAX_INPUT_TOKENS).sum(),
          "| their token estimate:", sorted(set(int(t) for t in long_tokens if t > MAX_INPUT_TOKENS)))
    totals = {"email": 0, "phone": 0, "order_id": 0}
    left_over = 0
    for text in tickets["text"]:
        safe_text, counts = redact(text)
        for name in counts:
            totals[name] += counts[name]
        left_over += sum(len(re.findall(pattern, safe_text)) for _, pattern, _ in PATTERNS)
    print("redactions:", totals, "| total:", sum(totals.values()), "| left after redaction:", left_over)

    print("\n=== Tasks 4-6: zero-shot run (retry log shown) ===")
    zero, zero_summary = run_pipeline(ZERO_SHOT)
    zero_classified = report("zero-shot", zero, zero_summary)
    print("zero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(zero_classified["true_category"], zero_classified["category"]))

    print("\n=== Task 6: few-shot run ===")
    few, few_summary = run_pipeline(FEW_SHOT, show_retries=False)
    report("few-shot", few, few_summary)

    print("\n=== Task 7a: token budget demo (zero-shot, TOKEN_BUDGET = 6000) ===")
    demo, _ = run_pipeline(ZERO_SHOT, token_budget=6000, show_retries=False)
    demo[columns].to_csv("outputs/triage_budget_demo.csv", index=False)
    print("tickets ending budget_exhausted:", (demo["status"] == "budget_exhausted").sum())
    print("zero-shot prompt length:", mock_llm.est_tokens(ZERO_SHOT), "estimated tokens")

    print("\n=== Task 7b: final run (few-shot, TOKEN_BUDGET = 100000) ===")
    final, _ = run_pipeline(FEW_SHOT, token_budget=100000, show_retries=False)
    final[columns].to_csv("outputs/triage_results.csv", index=False)
    print("rows:", len(final))
    print("routes:", final["route"].value_counts().to_dict())
    print("saved outputs/triage_results.csv and outputs/triage_budget_demo.csv")


if __name__ == "__main__":
    main()
