# src/policy_rag.py (Part D, tasks 1-4): policy question answering with RAG.
# RAG = Retrieval-Augmented Generation: first RETRIEVE matching policy text, then let the LLM answer from it.
# Run from the main folder:  python -m src.policy_rag

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)

POLICY_FOLDER = "data/policies/"
SYSTEM_PROMPT = ("POLICY ASSISTANT: answer only from the supplied context. If the answer is not in the "
                 "context, say you could not find the information in the policy documents.")


# ---------------- Task 1: chunking ----------------

def make_chunks(chunk_size=40, overlap=10):
    chunks = []
    step = chunk_size - overlap                      # 40 - 10 = each chunk starts 30 words later
    for file_name in sorted(os.listdir(POLICY_FOLDER)):
        words = open(POLICY_FOLDER + file_name, encoding="utf-8").read().split()
        start = 0
        while True:
            chunk_words = words[start:start + chunk_size]
            chunks.append({"source": file_name, "text": " ".join(chunk_words)})
            if start + chunk_size >= len(words):     # this chunk reached the last word: stop
                break
            start = start + step
    return chunks


# ---------------- Task 2: vectorise and retrieve ----------------

class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        self.chunks = make_chunks(chunk_size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])          # same fitted vectorizer
        scores = cosine_similarity(query_vector, self.chunk_vectors)[0]
        best = scores.argsort()[::-1][:k]                           # positions of the k highest scores
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 3),
                 "text": self.chunks[i]["text"]} for i in best]

    # ---------------- Task 3: augment and answer ----------------

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join("[" + hit["source"] + "]\n" + hit["text"] for hit in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "CONTEXT:\n" + context + "\nQUESTION:\n" + question},
        ]
        response = mock_llm.chat(messages)
        sources = []
        for hit in hits:
            if hit["source"] not in sources:
                sources.append(hit["source"])
        return {"answer": response.choices[0].message.content, "sources": sources}


# ---------------- Task 4: evaluation ----------------

def evaluate(rag):
    questions = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for _, q in questions.iterrows():
        result = rag.answer(q["question"])
        answerable = q["expected_doc"] != "NONE"
        rows.append({
            "question": q["question"],
            "answerable": answerable,
            "doc_retrieved": answerable and q["expected_doc"] in result["sources"],
            "fact_found": answerable and q["expected_fact"] in result["answer"],   # case-sensitive
            "fallback": result["answer"].startswith("I could not find"),
            "answer": result["answer"],
        })
    table = pd.DataFrame(rows)
    answerable = table[table["answerable"]]
    unanswerable = table[~table["answerable"]]
    summary = {
        "expected doc in top 3": f"{answerable['doc_retrieved'].sum()}/{len(answerable)}",
        "answer contains expected fact": f"{answerable['fact_found'].sum()}/{len(answerable)}",
        "fallback on unanswerable": f"{unanswerable['fallback'].sum()}/{len(unanswerable)}",
    }
    return table, summary


def main():
    rag = PolicyRAG(chunk_size=40, overlap=10)
    print("=== Task 1: chunks ===")
    print("total chunks:", len(rag.chunks))
    print(pd.Series([c["source"] for c in rag.chunks]).value_counts().to_dict())

    print("\n=== Task 2: retrieve('refund policy late delivery') ===")
    for hit in rag.retrieve("refund policy late delivery"):
        print(hit["source"], hit["score"])

    print("\n=== Task 3: example answer ===")
    print(rag.answer("What delivery time makes a completed order eligible for a full refund?"))

    print("\n=== Task 4: evaluation with overlap 10 ===")
    table, summary = evaluate(rag)
    print(summary)

    print("\n=== Task 4: evaluation with overlap 0 ===")
    rag_no_overlap = PolicyRAG(chunk_size=40, overlap=0)
    print("total chunks:", len(rag_no_overlap.chunks))
    table0, summary0 = evaluate(rag_no_overlap)
    print(summary0)
    broken = table0[table0["answerable"] & ~(table0["doc_retrieved"] & table0["fact_found"])]
    print("questions that break without overlap:")
    print(broken[["question", "answer"]].to_string(index=False))


if __name__ == "__main__":
    main()
