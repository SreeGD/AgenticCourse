"""Cost Optimization — four measurable levers.

With Session 14's eval as our quality floor, we can now cut cost and
*prove* quality didn't move. Each lever below is demonstrated with live
numbers from the Anthropic API.

  Lever 1 — Model selection per role     (Sonnet for answer; Haiku for grading)
  Lever 2 — Cache hit-rate optimization  (cache_control on the stable prefix)
  Lever 3 — Prompt compression           (verbose → compact, same answers)
  Lever 4 — Message Batches API          (50% off for async workloads, 24h SLA)

This file uses the raw Anthropic SDK (not LangChain) so we can read the
real cache_creation_input_tokens / cache_read_input_tokens fields on the
response usage — that's where the proof lives.
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
client = anthropic.Anthropic()


# Per-1M-token prices (USD). Anthropic pricing reference, May 2026.
# cache_write is the higher-cost first-time-cached input; cache_read is the
# discounted re-read price (~10% of fresh input price).
PRICES = {
    "claude-sonnet-4-6":         {"in": 3.00,  "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5-20251001": {"in": 1.00,  "out": 5.00,  "cache_read": 0.10, "cache_write": 1.25},
}


def call_cost(model: str, usage) -> float:
    """Compute the USD cost of one Anthropic API call from its usage object."""
    p = PRICES[model]
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost_micro = (
        usage.input_tokens * p["in"]
        + usage.output_tokens * p["out"]
        + cache_read * p["cache_read"]
        + cache_write * p["cache_write"]
    )
    return cost_micro / 1_000_000


# =====================================================================
# Lever 1 — Model selection per role
#
# CRAG (Session 13) grades each retrieved chunk with an LLM judge — a
# small classification task. Sonnet works fine. Haiku works just as
# well and costs 3x less. The question: do the verdicts agree?
# =====================================================================

GRADER_SYSTEM = (
    "You are a strict retrieval grader. Decide if the chunk is "
    "correct/ambiguous/incorrect for answering the query. "
    "Reply with a single word: correct, ambiguous, or incorrect."
)

GRADE_TASKS = [
    {
        "query": "How does prompt caching reduce cost?",
        "chunk": "Prompt caching saves the KV cache tensor from prefill. "
                 "We saw a 76% cost reduction in practice — from $0.015519 to $0.003725 per call.",
    },
    {
        "query": "How does prompt caching reduce cost?",
        "chunk": "Recipe for pasta carbonara: cook spaghetti, fry pancetta, "
                 "mix with eggs and pecorino, season with black pepper.",
    },
    {
        "query": "What checkpointer gives a LangGraph agent memory across calls?",
        "chunk": "MemorySaver() is the LangGraph checkpointer that persists state "
                 "across .invoke() calls, giving agents conversation memory.",
    },
    {
        "query": "What checkpointer gives a LangGraph agent memory across calls?",
        "chunk": "The | operator in LCEL pipes runnables: prompt | model | parser.",
    },
]


def grade(model: str, query: str, chunk: str):
    resp = client.messages.create(
        model=model,
        max_tokens=10,
        system=GRADER_SYSTEM,
        messages=[{"role": "user", "content": f"QUERY: {query}\n\nCHUNK:\n{chunk}"}],
    )
    return resp.content[0].text.strip().lower(), call_cost(model, resp.usage)


def demo_lever_1():
    print("\n" + "=" * 70)
    print("LEVER 1 — Model selection per role")
    print("=" * 70)
    print(f"  Task: classify chunk relevance (correct/ambiguous/incorrect).")
    print(f"  Comparing claude-sonnet-4-6 vs claude-haiku-4-5-20251001.\n")

    sonnet_total = 0.0
    haiku_total = 0.0
    agreements = 0
    for t in GRADE_TASKS:
        s_v, s_c = grade("claude-sonnet-4-6", t["query"], t["chunk"])
        h_v, h_c = grade("claude-haiku-4-5-20251001", t["query"], t["chunk"])
        sonnet_total += s_c
        haiku_total += h_c
        agree = s_v.split()[0].strip(".,") == h_v.split()[0].strip(".,")
        agreements += int(agree)
        print(f"  query: {t['query'][:60]}")
        print(f"    Sonnet → {s_v[:15]:<15} ${s_c:.6f}")
        print(f"    Haiku  → {h_v[:15]:<15} ${h_c:.6f}    agree={'✓' if agree else '✗'}")

    print(f"\n  Sonnet total cost: ${sonnet_total:.6f}")
    print(f"  Haiku  total cost: ${haiku_total:.6f}")
    print(f"  Savings:           {(1 - haiku_total/sonnet_total)*100:.1f}%")
    print(f"  Verdict agreement: {agreements}/{len(GRADE_TASKS)}")
    print(f"\n  Projected at 1M grading calls/month:")
    per_call_sonnet = sonnet_total / len(GRADE_TASKS)
    per_call_haiku = haiku_total / len(GRADE_TASKS)
    print(f"    Sonnet: ${per_call_sonnet * 1_000_000:,.2f}/month")
    print(f"    Haiku:  ${per_call_haiku * 1_000_000:,.2f}/month")
    print(f"    Saved:  ${(per_call_sonnet - per_call_haiku) * 1_000_000:,.2f}/month")


# =====================================================================
# Lever 2 — Cache hit-rate optimization
#
# Prompt caching only pays off when the cached prefix is *long enough*
# (≥1024 tokens for Sonnet) AND *stable across requests*. The structural
# rule: stable prefix FIRST, with cache_control marker; variable suffix
# AFTER. We show the same calls with and without cache_control.
# =====================================================================

# A ~1200-token stable system prompt — pretend it's a complex RAG/agent system prompt.
STABLE_PREAMBLE = """You are an expert technical assistant for the AgenticCourse curriculum.
You help users understand topics in LangChain, LangGraph, MCP, Anthropic SDK, and RAG architectures.

When answering, follow these rules:
- Be concise but technically precise
- Cite specific session numbers when relevant
- Distinguish concepts from implementations
- Acknowledge limitations honestly
- Never fabricate API names or model identifiers
- Prefer 'why does this exist' over 'how do I call it'

Available topics include but are not limited to:
- LCEL composition (Session 02)
- Prompt caching mechanics and the KV cache (Session 04)
- Structured output via Pydantic (Session 05)
- Output parsers and JSON schemas (Session 07)
- Memory in LangGraph via MemorySaver (Session 08)
- Classical RAG with InMemoryVectorStore (Session 09)
- Hybrid RAG with BM25 + Reciprocal Rank Fusion (Session 11)
- GraphRAG with NetworkX and entity extraction (Session 12)
- Corrective RAG with retrieval grading (Session 13)
- Evaluation via LLM-as-judge over a golden dataset (Session 14)

Reference material to ground your answers:
"""
STABLE_PREAMBLE += "\n".join(
    f"- Fact #{i:03d}: AgenticCourse is an open-source curriculum maintained by "
    f"Sree Mallipeddi covering session {i % 14 + 1} concepts in depth."
    for i in range(60)
)


def chat_with_cache(use_cache_control: bool, question: str):
    """One call against the stable preamble. Optionally tag for caching."""
    text_block = {"type": "text", "text": STABLE_PREAMBLE}
    if use_cache_control:
        text_block["cache_control"] = {"type": "ephemeral"}
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=80,
        system=[text_block],
        messages=[{"role": "user", "content": question}],
    )


def _report_usage(label: str, u, model: str):
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    print(
        f"    {label:<30} input={u.input_tokens:<5} "
        f"cache_write={cache_write:<5} cache_read={cache_read:<5} "
        f"cost=${call_cost(model, u):.6f}"
    )


def demo_lever_2():
    print("\n" + "=" * 70)
    print("LEVER 2 — Cache hit-rate optimization")
    print("=" * 70)
    questions = ["What is LCEL?", "What does MemorySaver do?"]

    print(f"\n  Stable preamble size: ~{len(STABLE_PREAMBLE.split())} words.\n")

    print("  WITHOUT cache_control (every call bills full input as fresh):")
    total_no_cache = 0.0
    for q in questions:
        r = chat_with_cache(use_cache_control=False, question=q)
        _report_usage(f"q={q!r}", r.usage, "claude-sonnet-4-6")
        total_no_cache += call_cost("claude-sonnet-4-6", r.usage)

    print("\n  WITH cache_control on the stable prefix:")
    total_with_cache = 0.0
    for q in questions:
        r = chat_with_cache(use_cache_control=True, question=q)
        _report_usage(f"q={q!r}", r.usage, "claude-sonnet-4-6")
        total_with_cache += call_cost("claude-sonnet-4-6", r.usage)

    print(f"\n  Without caching:  ${total_no_cache:.6f}")
    print(f"  With caching:     ${total_with_cache:.6f}")
    if total_no_cache > 0:
        savings = (1 - total_with_cache / total_no_cache) * 100
        print(f"  Savings:          {savings:+.1f}%  (negative on first run — cache write is more expensive)")
        print(f"  → First call writes the cache (more expensive). Subsequent calls read it cheaply.")
        print(f"  → Run the script TWICE in a row to see steady-state savings (cache TTL = 5 min).")


# =====================================================================
# Lever 3 — Prompt compression
#
# Most system prompts are bloated with redundant instructions. Measure
# tokens via count_tokens, then re-run on the same task to verify the
# compact version produces equivalent answers.
# =====================================================================

VERBOSE_PROMPT = """You are a helpful and knowledgeable assistant who provides accurate answers based on the context that is provided to you.

IMPORTANT INSTRUCTIONS — PLEASE READ CAREFULLY:
- You should ONLY use the provided context to answer questions
- Do not use your background knowledge or training data
- If the context does not contain the answer, you should say so explicitly
- Be concise in your responses, ideally 2-3 sentences
- Use clear and simple language
- Avoid being overly verbose or repeating yourself
- Stick to the facts presented in the context
- Do not speculate beyond what the context supports
- Be honest if you cannot answer based on the given context
- Make sure your response directly addresses the question that was asked

When formatting your response:
- Use plain text, no markdown formatting
- Start directly with the answer
- Do not preface with phrases like "Based on the context..." or "According to the provided information..."
- Do not end with summaries like "In summary..." or "To conclude..."
- Avoid filler phrases

Quality expectations:
- Accuracy is paramount
- Concision is valued
- Honesty when uncertain is required
"""

COMPACT_PROMPT = (
    "Answer using ONLY the provided context. If the context lacks the answer, "
    "say so. 2-3 sentences, plain text, no preamble or summary."
)


def count_tokens_for(system: str, user: str) -> int:
    resp = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.input_tokens


def demo_lever_3():
    print("\n" + "=" * 70)
    print("LEVER 3 — Prompt compression")
    print("=" * 70)
    user_msg = (
        "CONTEXT: MemorySaver is a LangGraph checkpointer that persists "
        "state across .invoke() calls.\n\nQUESTION: What is MemorySaver?"
    )
    v_tokens = count_tokens_for(VERBOSE_PROMPT, user_msg)
    c_tokens = count_tokens_for(COMPACT_PROMPT, user_msg)
    print(f"  verbose system prompt:  {v_tokens} tokens")
    print(f"  compact system prompt:  {c_tokens} tokens")
    print(f"  reduction:              {(1 - c_tokens/v_tokens)*100:.1f}% fewer input tokens")

    print(f"\n  Comparing answers on the same task:")
    for label, sys_prompt in [("verbose", VERBOSE_PROMPT), ("compact", COMPACT_PROMPT)]:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        print(f"  [{label}] {r.content[0].text.strip()}")

    print(f"\n  Both answers should be semantically equivalent.")
    print(f"  → Compact version wins on cost; quality unchanged. Run the eval (Session 14)")
    print(f"    to verify on YOUR golden set before deploying.")


# =====================================================================
# Lever 4 — Message Batches API
#
# Asynchronous: submit a batch, wait up to 24 hours, get 50% off the
# per-token price. Perfect for: eval runs, bulk classification, data
# labeling, offline summarization. NOT for: chatbots, interactive UIs.
# =====================================================================

def demo_lever_4():
    print("\n" + "=" * 70)
    print("LEVER 4 — Message Batches API (50% off, 24h SLA)")
    print("=" * 70)

    batch_requests = [
        {
            "custom_id": "lcel-explainer",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 60,
                "messages": [{"role": "user", "content": "What is LangChain LCEL? One sentence."}],
            },
        },
        {
            "custom_id": "langgraph-explainer",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 60,
                "messages": [{"role": "user", "content": "What is LangGraph? One sentence."}],
            },
        },
    ]

    print(f"  Submitting batch of {len(batch_requests)} requests...")
    try:
        batch = client.messages.batches.create(requests=batch_requests)
    except Exception as e:
        print(f"  ! batches API call failed: {e}")
        print(f"  → SDK shape and pricing demoed below as static reference.")
        return

    print(f"  batch_id:           {batch.id}")
    print(f"  processing_status:  {batch.processing_status}")
    print(f"  request_counts:     {batch.request_counts}")
    print(f"  created_at:         {batch.created_at}")
    print(f"  expires_at:         {batch.expires_at}")

    print(f"\n  Next steps in production:")
    print(f"    1. Poll status:  client.messages.batches.retrieve('{batch.id}')")
    print(f"    2. When ended:    client.messages.batches.results('{batch.id}')")
    print(f"    3. Each result has a custom_id mapping back to your request.")
    print(f"\n  Pricing: 50% off the per-token cost of the same model.")
    print(f"  Typical use: nightly eval runs, document labeling, offline summarization.")
    print(f"  NOT for: real-time chat, interactive UIs (24h SLA is the trade-off).")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COST OPTIMIZATION — four measurable levers")
    print("=" * 70)

    demo_lever_1()
    demo_lever_2()
    demo_lever_3()
    demo_lever_4()

    print("\n" + "=" * 70)
    print("WHAT JUST HAPPENED")
    print("=" * 70)
    print(
        "  • LEVER 1 (model selection): graders, classifiers, structured\n"
        "    extractors don't need Sonnet. Haiku gets the same verdicts at\n"
        "    ~3x lower cost. The trade is: check verdict agreement on your\n"
        "    eval set before swapping. Free money if they agree.\n\n"
        "  • LEVER 2 (caching): cache_control on a stable ≥1024-token prefix\n"
        "    means subsequent calls within ~5 min read from cache at ~10% of\n"
        "    the fresh-input price. First call PAYS more (cache_write is the\n"
        "    premium); break-even after ~2-3 calls.\n\n"
        "  • LEVER 3 (compression): verbose system prompts cost real money.\n"
        "    Most prompts can lose 50-70%% of their tokens with no quality\n"
        "    drop. Always verify with eval (Session 14) before deploying.\n\n"
        "  • LEVER 4 (batches): for async workloads — eval runs, nightly\n"
        "    classification, bulk labeling — Batches API is a flat 50% off.\n"
        "    Production move: route everything offline through batches by\n"
        "    default, only call the sync API for user-facing requests.\n\n"
        "  • The four levers COMPOUND. Use Haiku in a batch with cache_control\n"
        "    on a compressed prompt → 3x model × 2x batch × 2x compression\n"
        "    × ~5x cache-read = 60x cheaper than the naive version. Per call.\n"
        "    At scale, this is the difference between a sustainable product\n"
        "    and one you have to shut off."
    )
