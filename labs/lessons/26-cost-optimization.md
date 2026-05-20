# 26 — Cost Optimization (Session 15)

> **The four levers that compound.** Model selection per role, cache hit-rate optimization, prompt compression, Batches API. Each is measurable. With Session 14's eval as the quality floor, you can pull each lever and *prove* quality didn't move. Stack them and the savings multiply — a naive setup costs 60x more than the optimized one. Per call.

---

## Roadmap — where this lesson sits in the journey

```
═══════ PHASE 1: FOUNDATION (done) ═══════                ═══════ PHASE 2 ═══════

  ✓ 01-14 (foundation + RAG + eval)                        Track F: PRODUCTION
                                                             ✓ Session 14: Evaluation
                                                             ▶ Session 15: COST OPTIMIZATION  ◄ HERE
                                                             ○ Session 16: Streaming
                                                             ○ Session 17: Deploy + Observability
                                                           Track G: ○ Architect Skills
```

**Why this lesson now:** You can't claim a cost reduction is "free" without showing eval scores didn't move. Session 14 gave us that gate. Now we wield it.

---

## File involved

| File | Role |
|---|---|
| [`26_cost_optimization.py`](../26_cost_optimization.py) | All four levers in one runnable file, using the raw Anthropic SDK so you can read `cache_creation_input_tokens` / `cache_read_input_tokens` directly off the response. |

This file does **not** use LangChain because we need the raw `usage` object — LangChain abstracts those numbers away. For production code you'd put these patterns behind your own LangChain wrappers; for *measuring*, go raw.

---

## What problem it solves

Most AI cost blowups have the same shape:
- Expensive model used for cheap roles (Sonnet/Opus grading binary classifications)
- Cache hit rate of 0% because no one marked the stable prefix
- System prompts bloated with redundant instructions copied from a tutorial
- Everything running synchronously because no one knew batches existed

Each is a measurable, fixable mistake. This lesson is the operating manual for fixing all four. The cost savings on real workloads are often **5x to 60x** with **zero quality impact** — provable via the eval harness from Session 14.

---

## The analogy

**A power bill audit.**

Most companies pay 2-5x what they should for electricity because nobody ever audited:
- Are we on the right tariff? (model selection)
- Are we running heaters and AC simultaneously? (cache miss = paying twice for the same prefix)
- Are we leaving lights on in empty rooms? (verbose prompts = paying for tokens nobody reads)
- Are we running non-urgent loads during peak hours? (sync API for offline workloads)

Fix the four and the bill drops by an order of magnitude. Same building, same usage pattern, just stopped paying for the same thing four ways.

---

## Visual

```
        NAIVE SETUP                              OPTIMIZED SETUP
        ───────────                              ───────────────

   ┌──────────────────┐                     ┌──────────────────┐
   │ Sonnet for every │                     │ Haiku for cheap  │   ← Lever 1
   │ role             │                     │ roles            │     3x cheaper
   └─────────┬────────┘                     └─────────┬────────┘
             │                                        │
             ▼                                        ▼
   ┌──────────────────┐                     ┌──────────────────┐
   │ No cache_control │                     │ cache_control on │   ← Lever 2
   │ ⇒ 0% hit rate    │                     │ stable prefix    │     ~5x on hits
   └─────────┬────────┘                     └─────────┬────────┘
             │                                        │
             ▼                                        ▼
   ┌──────────────────┐                     ┌──────────────────┐
   │ Verbose prompt   │                     │ Compressed       │   ← Lever 3
   │ 300+ tokens of   │                     │ prompt           │     ~2-3x fewer
   │ filler           │                     │ 80 tokens        │     input tokens
   └─────────┬────────┘                     └─────────┬────────┘
             │                                        │
             ▼                                        ▼
   ┌──────────────────┐                     ┌──────────────────┐
   │ Sync API for     │                     │ Batches API for  │   ← Lever 4
   │ everything       │                     │ offline workload │     2x cheaper
   └─────────┬────────┘                     └─────────┬────────┘
             │                                        │
             ▼                                        ▼
       ≈ $X/month                            ≈ $X/60 month
```

---

## Concept walk-through

### Lever 1 — Model selection per role

Not every LLM call is the same task.

| Role | Task type | Right model |
|---|---|---|
| Final answer to user | Generation, reasoning | Sonnet / Opus |
| Retrieval grader (CRAG) | Ternary classification | **Haiku** |
| Output classifier | Binary classification | **Haiku** |
| Structured extraction (entities) | JSON shape generation | **Haiku** |
| Query rewriter | Short text generation | **Haiku** |
| Reflection / planner | Multi-step reasoning | Sonnet |
| Tool argument generation | Structured output | Sonnet (sometimes Haiku) |

The rule: **use the smallest model that gets the right answer on your eval set.** Sonnet is needed for the hard parts. Haiku handles the rest at ~3x less.

From the live run:
```
query: How does prompt caching reduce cost?
  Sonnet → correct      $0.000381
  Haiku  → correct      $0.000126    agree=✓
```
Same verdict. 3x cheaper. Free money once you've verified agreement on your set.

Projected at 1M grading calls/month:
- Sonnet: $353.25
- Haiku: $116.75
- **Saved: $236.50/month**

For ONE component. A real RAG pipeline has 3-5 of these.

### Lever 2 — Cache hit-rate optimization

Anthropic charges three different rates for input tokens:
- **Fresh input**: $3.00/M (Sonnet)
- **Cache write**: $3.75/M (the *first* time a prefix is cached — costs 25% MORE than fresh)
- **Cache read**: $0.30/M (every *subsequent* call within ~5 min — 90% off fresh)

The trick: tag your stable prefix with `cache_control: ephemeral`. Anthropic stores the KV cache server-side for ~5 minutes; subsequent calls that share the prefix read from cache.

```python
client.messages.create(
    model="claude-sonnet-4-6",
    system=[
        {
            "type": "text",
            "text": LONG_STABLE_PROMPT,            # ≥1024 tokens
            "cache_control": {"type": "ephemeral"} # ← the magic line
        }
    ],
    messages=[{"role": "user", "content": user_question}],
)
```

From the live run, second call within seconds of the first:
```
WITHOUT cache_control:
  q='What is LCEL?'             input=2476  cache_write=0     cache_read=0     cost=$0.008628
  q='What does MemorySaver do?' input=2479  cache_write=0     cache_read=0     cost=$0.008637

WITH cache_control:
  q='What is LCEL?'             input=3     cache_write=2473  cache_read=0     cost=$0.010483
  q='What does MemorySaver do?' input=3     cache_write=12    cache_read=2464  cost=$0.001993
```

The first call with `cache_control` paid MORE ($0.010 vs $0.0086) — cache writes are premium. The **second** call read 2464 cached tokens at $0.30/M and dropped to $0.0020 — 4x cheaper than uncached.

**Structural rules for cache hits:**
1. Stable prefix FIRST, variable suffix LAST. (Cache only works for prefix matches.)
2. Prefix must be ≥1024 tokens for Sonnet (lower for Haiku).
3. The text must be **byte-exact** across calls. Adding even one different word breaks the cache.
4. TTL is ~5 minutes. After that, the next call writes a fresh cache.

### Lever 3 — Prompt compression

Most system prompts are bloated tutorial templates. The pattern:

**Before** (302 tokens):
```
You are a helpful and knowledgeable assistant who provides accurate answers...
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
- Do not preface with phrases like "Based on the context..."
...
```

**After** (81 tokens):
```
Answer using ONLY the provided context. If the context lacks the answer,
say so. 2-3 sentences, plain text, no preamble or summary.
```

73% fewer tokens. Identical answers in the live run:
```
[verbose] MemorySaver is a LangGraph checkpointer that persists state across .invoke() calls.
[compact] MemorySaver is a LangGraph checkpointer that persists state across .invoke() calls.
```

**Compression rules:**
1. Strip filler ("please read carefully", "make sure")
2. Collapse multi-sentence rules into one sentence with semicolons
3. Drop "do not" lists when an affirmative covers them ("plain text" replaces "no markdown", "no formatting")
4. Drop politeness ("kindly", "if you would")
5. Verify with eval. **Always.** Compression that drops quality is just damage.

This works because LLMs already know how to answer. Prompts mostly *constrain* output — they don't *teach* the task. The minimum constraint is usually enough.

### Lever 4 — Message Batches API

For asynchronous workloads (eval runs, nightly classification, bulk summarization), the Batches API gives a flat **50% discount** in exchange for a 24-hour SLA.

```python
batch = client.messages.batches.create(requests=[
    {
        "custom_id": "q1",
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 60,
            "messages": [{"role": "user", "content": "..."}],
        },
    },
    # ... up to 100,000 requests per batch
])

# returns immediately with batch.id and processing_status="in_progress"
# poll via:
status = client.messages.batches.retrieve(batch.id)
# when status.processing_status == "ended":
results = client.messages.batches.results(batch.id)
# each result has its custom_id back so you can match them up.
```

From the live run, an actual batch submission:
```
batch_id:           msgbatch_013FPered6AFtZvoNzQ2ofJ9
processing_status:  in_progress
created_at:         2026-05-19 17:37:59
expires_at:         2026-05-20 17:37:59    (24 hours later)
```

**Use Batches for:**
- Nightly eval runs (your Session 14 harness — fits perfectly)
- Bulk document classification
- Offline data labeling pipelines
- Backfill summarization

**Don't use Batches for:**
- Chatbots (24h SLA breaks UX)
- Anything user-facing in real time
- Workloads under ~100 requests (batch overhead isn't worth it)

---

## Run it

```
cd labs
./.venv/bin/python 26_cost_optimization.py
```

Takes ~45 seconds, costs ~$0.05 total. Important: **run it twice in a row** to see Lever 2 in steady state — the second run will hit the cache from the first run if you do it within 5 minutes.

---

## Real output highlights

**Lever 1 — Model swap savings:**
```
Sonnet total cost: $0.001413
Haiku  total cost: $0.000467
Savings:           66.9%
Verdict agreement: 4/4

Projected at 1M grading calls/month:
  Sonnet: $353.25/month
  Haiku:  $116.75/month
  Saved:  $236.50/month
```

**Lever 2 — Cache proof:**
```
WITH cache_control, second call:
  q='What does MemorySaver do?'  input=3     cache_write=12   cache_read=2464   cost=$0.001993
```
2464 tokens read from cache at $0.30/M = the deal you signed up for.

**Lever 3 — Compression:**
```
verbose system prompt:  302 tokens
compact system prompt:  81 tokens
reduction:              73.2% fewer input tokens
```
Both produced identical answers.

**Lever 4 — Real batch:**
```
batch_id:  msgbatch_013FPered6AFtZvoNzQ2ofJ9
expires_at: 2026-05-20 17:37:59+00:00
```

---

## Production patterns

### The compound effect

These levers stack multiplicatively, not additively:

| Stack | Multiplier on naive |
|---|---|
| Model: Sonnet → Haiku | 3x |
| Caching: 0% hits → 90% hits on a 2000-token prefix | ~4x |
| Compression: 300-token prompt → 80-token prompt | ~2x |
| Async path moved to Batches | 2x |
| **Compound** | **~50-60x** |

A naive RAG pipeline that costs $1.00/query becomes ~$0.02/query. **Same quality.** Verifiable via eval.

### Decision flow for a new workload

```
1. Is this user-facing realtime?
   YES → sync API
   NO  → ALWAYS Batches (50% off, free)

2. Is this a generation/reasoning task or a classification/extraction?
   GEN   → Sonnet, sometimes Opus
   CLASS → Haiku (verify on eval first)

3. Does your prompt have a stable prefix ≥1024 tokens?
   YES → cache_control on it
   NO  → either pad to make it cacheable, or live with cache misses

4. Has your system prompt been reviewed for compression in the last 6 months?
   NO  → review it. Target 50%+ reduction. Verify on eval.
```

### Where compression bites you

- **Few-shot examples**: removing them often drops quality on edge cases. Verify carefully.
- **Format constraints**: "respond in JSON" can't be compressed below itself.
- **Persona / tone instructions**: "you are friendly" is two tokens; can't be shorter.
- **Multi-turn agents**: the agent's tools description is hard to compress without losing precision.

### Cache TTL gotcha

The default 5-minute TTL works for chatbot sessions where a user keeps typing. It does NOT work for:
- Cold starts (every Lambda invocation pays cache write)
- Batched eval runs (calls spaced > 5 min apart get no cache benefit)

Anthropic also offers **1-hour TTL** caching (`cache_control: {"type": "ephemeral", "ttl": "1h"}`) for an extra 2x premium on the write but the same read rate. Worth it if your traffic pattern is bursty over the hour.

### Monitoring cache hit rate

Log `cache_read_input_tokens / total_input_tokens` per call. Aggregate by endpoint. Target: **>60% hit rate on chatbot endpoints**, **>90% on RAG endpoints with a stable corpus prefix**. If you're below those, your prefix isn't byte-exact across calls — probably a timestamp or session ID leaking in.

### Routing offline through Batches

The cleanest production pattern: route ALL non-realtime work through Batches by default. Have one sync code path for user-facing requests, one batch code path for everything else. Eval, labeling, embedding refresh, summarization — all batch.

Cost win on Lever 4 alone, for a typical product that's 80% async/20% sync workload: ~40% of total API spend, automatic.

---

## Try this

1. **Switch the Session 14 eval grader to Haiku.** Re-run `25_evaluation.py` with `model = ChatAnthropic(model="claude-haiku-4-5-20251001")`. Compare scores against the Sonnet baseline. If they agree within ±0.05, congratulations — you just cut eval cost by 3x.

2. **Audit your own Claude Code sessions.** Look at the system prompt that gets sent for an agent loop. Count its tokens. Now compress it. Run the same task and see if behavior changes.

3. **Push cache TTL to 1 hour.** Add `"ttl": "1h"` to the cache_control block in Lever 2. Re-run the script twice with > 5 min between runs. Verify the second run still gets cache hits.

4. **Move your eval pipeline to Batches.** Wrap the Session 14 eval to submit all judge calls as a single batch. The eval will take ~24h to complete, but cost will drop 50% — perfect for nightly CI.

5. **Build a cost dashboard.** For every Anthropic call in your app, log `model`, `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `endpoint`. Aggregate daily. Find your biggest line item — that's your next optimization target.

---

## Mental model

> **Each lever costs zero engineering hours once the discipline is in place. The compound savings are real. The eval gate is the only thing standing between "I think we can swap to Haiku" and "the harness proved we can swap to Haiku."**

Cost optimization is not a feature you build. It is a *discipline* you adopt:

1. Before every prompt: which role is this? Pick the right model.
2. Before every prompt: is there a stable prefix? Mark it.
3. Before every prompt: is every word earning its place? Trim it.
4. Before every workload: does it need to be sync? Push it to batches.

Do this for a week and your bill drops by an order of magnitude. Permanently.

---

## FAQ

**Q: Is Haiku really enough for graders?**
For ternary/binary classification with a clear rubric — yes, usually. Failure mode is when the rubric requires nuanced judgment (subtle hallucination detection, multi-step reasoning). Always verify on your eval set first. Some teams use Haiku for the grader, Sonnet for the final answer — best of both.

**Q: What if my prompts change between calls?**
Cache is byte-exact prefix matching. If your stable prefix is genuinely stable, you're fine. If session IDs or timestamps leak in, you'll see cache_creation_input_tokens on every call (which is the symptom of "the cache didn't catch anything").

**Q: Should I cache the user message too?**
Usually no — user messages are the *variable* part. The cache_control marker goes on the **system** prompt (or the assistant's tool definitions, or any long content that's stable across calls).

**Q: How aggressive can prompt compression go?**
Until eval breaks. Some teams report 70-80% reduction with no quality drop on well-defined tasks. Open-ended generation tolerates less. The only way to know is to run the eval before and after.

**Q: Doesn't Batches API mean I have to refactor my code?**
Mostly no. Wrap your existing `messages.create()` calls in a helper that *also* knows how to submit them as batch entries. Sync path calls helper synchronously; batch path queues entries and calls helper at end-of-batch. Same call structure, different routing.

**Q: What about Anthropic's prompt caching with the new long-context model?**
Same mechanism. Long context (1M tokens with Claude Sonnet 4.6 1M) makes caching even MORE valuable — a million-token prefix uncached is $3.00. Cached read is $0.30. The compounding gets dramatic at scale.

**Q: How do these compose with rate limits / quotas?**
Cache reads count against input-token quota at the cache_read rate (which is lower in tokens-per-minute terms too). Batches have a separate, higher quota. Both effectively raise your throughput ceiling.

**Q: What's the breakeven for cache_control?**
The first call costs ~25% MORE (cache write premium). Each subsequent call within the TTL costs ~90% LESS. Breakeven: **~2 calls within the TTL window** is enough to come out ahead. For a chatbot session that has 5+ turns, caching is a no-brainer.

**Q: Can I cache prompts for fine-tuned models?**
No — caching is a Claude API feature, not a model feature. Works on all Claude API models. Fine-tuned models live on different infrastructure with their own caching story (which is generally weaker).

**Q: How does Lever 4 (Batches) interact with eval?**
Beautifully. Eval is the canonical async workload — you don't need results in real time, you just need them tonight before the next sprint review. Run your Session 14 eval via Batches and pay half. The only change is `client.messages.create(...)` becomes `client.messages.batches.create(requests=[...])`.

---

## Related

- **Previous:** [25 — Evaluation](25-evaluation.md) — the quality floor that makes every optimization safe
- **Next:** Session 16 — Streaming (latency optimization, the other half of UX)
- **Builds on:** [04 — Prompt caching](04-prompt-caching.md) (Lever 2 deep dive), [18 — Anthropic SDK](18-anthropic-sdk.md) (raw client used here), [24 — Corrective RAG](24-corrective-rag.md) (the grader is the prime Haiku-swap target)
- **Track F status:** ▶ 2/4 complete. Eval → Cost. Next: Streaming → Deploy.
