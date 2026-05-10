# LangChain — Part 2: Frameworks, Token Economics, and Prompt Caching

A second-half companion to [`NOTES.md`](./NOTES.md). That doc covered the foundations — `hello.py`, `chain.py`, `agent.py` — and ended with the most important mental model: *"The LLM never calls anything. The LLM Client does."*

This doc continues from there. We move from "I built an agent loop by hand" to "the framework version" to "I can see what every turn actually costs" to "I can make those turns 76% cheaper with one keyword."

**Files added in this part:**

| File | What it adds |
|---|---|
| `agent_lg.py` | The framework version of the manual loop in `agent.py` — uses `create_react_agent` from `langgraph` |
| `agent_lg_cached.py` | The same agent, with prompt caching wired up. Demonstrates the cost difference. |

---

## Step 4 — `agent_lg.py`: the framework version of the agent

### Same agent, different driver

`agent.py` had a hand-written `while True:` loop that:
1. Called `model.invoke(history)`
2. Checked for `tool_calls`
3. Ran each tool, appended `ToolMessage` results
4. Looped until no more tool calls

Now we replace all of that with **one function call**:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools=[add, get_current_time])
result = agent.invoke({"messages": [("user", "What's 47 + 158, and what time is it?")]})
print(result["messages"][-1].content)
```

That's the entire loop. Same propose → execute → feedback → answer dance, just inside a battle-tested implementation.

### Reading the trace

Use `stream_mode="values"` to see every intermediate state — same trace as `agent.py`'s manual loop:

```python
for event in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

You'll see exactly what you saw with `agent.py`:
```
Human Message
Ai Message     ← with tool_calls (no .content text)
Tool Message
Ai Message     ← final answer
```

### What you got for free

Beyond saving 17 lines of loop code, `create_react_agent` gives you:

| Feature | Without it | With it |
|---|---|---|
| Recursion limit | infinite-loop risk | enforced |
| Parallel tool exec | manual coordination | automatic |
| Streaming | rebuild each turn | `stream()` works out of the box |
| Checkpointing | impossible | one parameter |
| Human-in-the-loop interrupts | impossible | `interrupt_before=[...]` |
| `.batch()`, `.astream()` | rewrite | inherited from `Runnable` |

Each row is 20-100 lines you don't have to write — but more importantly, don't have to *get right*.

### The "messages" channel pattern

LangGraph state is keyed by **channels**. The default ReAct agent uses one channel called `messages`:

```python
agent.invoke({"messages": [("user", "...")]})  # input: a list keyed by "messages"
                ↓
    {"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage]}  # output: appended state
```

That dict-with-`messages` shape is the convention you'll see everywhere in LangGraph. When you build custom graphs later, you'll define your own channels — but `messages` is the universal one for ReAct-style agents.

---

## Step 5 — measuring what every turn costs

### The `usage_metadata` field

Every `AIMessage` from a tool-aware model carries token accounting in `usage_metadata`. LangChain normalizes the shape across providers:

```python
{
    "input_tokens": 640,
    "output_tokens": 102,
    "total_tokens": 742,
    "input_token_details": {
        "cache_read": 0,
        "cache_creation": 0,
    },
}
```

To track per-call and total cost across an agent run, walk the message list and accumulate:

```python
from langchain_core.messages import AIMessage

def total_usage(messages):
    return sum((m.usage_metadata or {}).get("input_tokens", 0)
               for m in messages if isinstance(m, AIMessage))
```

(`agent_lg.py` has a fuller `print_usage()` that breaks it down per call and surfaces cache fields.)

### The four lessons hidden in the numbers

When we ran `agent_lg.py`, we saw something like:

```
call 1: in= 640  out= 102  total=742
call 2: in= 818  out=  61  total=879
TOTAL:  in=1458  out= 163  total=1621
```

Four things that changed how I think about agent cost:

**1. Each turn carries the whole conversation.**
`call 2`'s 818 input tokens = `call 1`'s 640 + the AI's tool-call message + both tool results. The agent is *stateless from the model's perspective* — every turn re-sends the full history. A 10-turn agent doesn't pay for 10 calls; it pays for the running sum: `1+2+3+...+10 = 55` worth of growing context.

**2. Tool schemas aren't free.**
Most of `call 1`'s 640 input tokens is the **JSON schema** of the bound tools, not the user's question. Add 20 tools and you're paying thousands of tokens *before the user has typed anything*.

**3. Input is deterministic; output is noisy.**
At `temperature=0`, the input token count is byte-stable across runs (same prompt, same tools, same history). Output tokens vary by ±15 even at `temperature=0` — Claude phrases the same answer slightly differently each time. **Budget input as a constant; budget output as a distribution.**

**4. Per-token cost (Sonnet 4.6).**
- Input: $3 per 1M tokens
- Output: $15 per 1M tokens
- Output is **5× more expensive per token** than input. A verbose agent costs more from response length than from prompt size.

### Worked example: when does this matter?

A 5-turn agent run with our setup:

```
Turn 1:   600 in   →  100 out
Turn 2:   850 in   →   80 out
Turn 3:  1100 in   →   80 out
Turn 4:  1350 in   →   80 out
Turn 5:  1600 in   →  100 out
─────────────────
TOTAL:  5500 in   +  440 out
COST:   $0.0165   +  $0.0066   = ~$0.023 per session
```

Scale to 10k users × 10 sessions/day = **$2,300/day**. Token math becomes a P&L line very fast.

---

## Deep dive: who actually calls the tools?

This is the question that — once answered correctly — turns agents from magic into machinery.

### The wrong mental model

> "The LLM has access to my tools and decides which ones to call."

This is what every demo and most documentation implies. **It's wrong.** It causes confusion about security, debugging, latency, and how to scale.

### The right mental model

> **The LLM never calls anything. The LLM Client does.**

The LLM is a **stateless text-completion function** — text in, text out. It cannot:
- Execute code
- Hit a clock
- Open a network connection
- Read a file
- Spawn a subprocess

What it *can* do is **emit structured JSON** saying "I would like the function called `add` to be invoked with arguments `{a: 47, b: 158}`."

That's it. That's the whole "tool use" mechanism. The LLM proposes; your code disposes.

### What the LLM Client actually does

```
  +-----------+                       +-----------------+
  |   YOU     |                       |   Anthropic     |
  | (the LLM  |                       |    (Claude)     |
  |  Client)  |                       |                 |
  +-----+-----+                       +--------+--------+
        |                                      |
        |  prompt + tool schemas               |
        |------------------------------------->|
        |                                      |
        |     "please call add(47,158)"        |
        |<-------------------------------------|
        |                                      |
        |  *** YOU run add(47,158) → 205 ***   |
        |       (Claude has no idea            |
        |        this is happening)            |
        |                                      |
        |  prompt + history + result=205       |
        |------------------------------------->|
        |                                      |
        |     "47+158 is 205. ..."             |
        |<-------------------------------------|
        v                                      v
```

In `agent.py`, the LLM Client is your hand-written `while True:` loop.
In `agent_lg.py`, the LLM Client is `create_react_agent`.
In Cursor / Claude Desktop / Cline, the LLM Client is the IDE.
In OpenAI's Assistants API, the LLM Client is OpenAI's runtime.

It's **always something on your side of the wire.** The LLM is just the inference call in the middle.

### Why this design is the only sane one

| Property | Because the LLM doesn't actually run anything... |
|---|---|
| **Security** | The model can't `rm -rf /`. It can only ask. You decide whether to honor the request. |
| **Auditability** | Every action is your Python code. You log, rate-limit, redact, or refuse any call. |
| **Portability** | A "tool" can be a Python function, a SQL query, a REST call, a Bash command, an MCP server, a robot arm. The LLM doesn't know or care. |
| **Determinism where it matters** | Your `add` function is deterministic Python. Only the *decision* to call it is fuzzy LLM output. |

### Vocabulary, decoded

Once you have the "LLM Client" frame, the rest of the ecosystem maps cleanly:

| Term | What it really is |
|---|---|
| **Agent** | An LLM Client that runs the propose → execute → feedback loop |
| **Tool** | A function the LLM Client is willing to run on the LLM's behalf |
| **MCP server** | A remote toolbox; the LLM Client connects to it to discover and invoke tools |
| **MCP client / host** | The LLM Client, with the tool layer abstracted over a network protocol |
| **Multi-agent system** | Multiple LLM Clients (or one Client orchestrating multiple LLM personas) |
| **Guardrails / approvals** | Rules the LLM Client enforces *before* honoring a tool request |

---

## Step 6 — `agent_lg_cached.py`: prompt caching

### The problem in your own numbers

Look at what `agent_lg.py` showed:

```
call 1:   640 in
call 2:   818 in     ← 818 tokens, 100% of which Claude already saw on call 1
```

You're paying full price for tokens the model just processed. Multi-turn agents waste 60-90% of their input cost re-sending the same prefix.

### The fix in plain terms (the chef analogy)

Imagine Claude is a chef and your prompt is a 50-page recipe.

**Without caching:** every time you order, the chef reads all 50 pages, chops the vegetables, makes the sauce — *then* cooks your dish.

**With caching:** day 2, you hand over the same 50-page recipe. The chef recognizes it: *"Already chopped these vegetables yesterday, sauce is in the fridge, oven's preheated."* Skips straight to cooking.

You pay 10% for the prep (already done) and full price only for the new cooking.

### What's *actually* cached

The technical version: when the LLM processes your prompt during **prefill** (the prompt-ingestion phase before generation), it computes a giant tensor called the **KV cache** — the Key and Value projections at every layer for every token. For an N-token prompt, this is roughly:

```
80 layers × N × 64 heads × 128 dims × 2 (K and V) × 2 bytes (bf16)
= ~2.6 MB per token
```

That tensor is the *output* of the most expensive part of inference. Without prompt caching, it's recomputed from scratch every request and discarded.

**With `cache_control` enabled, Anthropic keeps that tensor alive between requests, indexed by a hash of the prompt prefix.** On a cache hit, the prefill forward pass for the cached portion is **skipped entirely** — replaced with a memory load.

### The wire reality (this trips everyone up)

> **The client always sends the full prompt. Every byte. Every time.**

`cache_control` is *not* a reference to cached content. It's a hint to Anthropic saying "please cache the prefix up to this marker." The client is stateless. The cache is server-side, transparent, content-addressed by hashing the prompt prefix.

| Question | Answer |
|---|---|
| Does my outbound bandwidth go down? | **No.** Same bytes on the wire. |
| Does latency go down? | **Yes.** Anthropic skips ~50-80% of prompt processing time. |
| Does my dollar cost go down? | **Yes.** Cached portion billed at 0.1× input rate. |
| Do I manage cache state in my client? | **No.** Just add `cache_control` markers. |
| Can I check whether a hit happened? | **Yes.** `usage_metadata.input_token_details.cache_read`. |

Better analogy than HTTP `If-None-Match`: it's a **CDN edge cache**. Same request, same bytes, same response — but the origin says "served this exact URL+headers 30 seconds ago, here's the cached compute" and bills accordingly.

---

## Why prompt caching costs less (mechanics, not marketing)

### Prefill vs decode — the two phases of LLM inference

```
PREFILL (prompt processing):
  - Process ALL N input tokens at once
  - For each layer (~80 in Sonnet), compute Q,K,V; do attention; do MLP
  - Output: the KV cache + next-token logits
  - Cost: ~O(N²), COMPUTE-bound (GPU FLOPs maxed)

DECODE (generation):
  - Produce one output token at a time
  - Each new token attends to all cached K/V
  - Cost: ~O(N) per token, MEMORY-BANDWIDTH-bound (FLOPs idle)
```

For a typical request, **prefill consumes 60-90% of the GPU work** even though it produces no visible output. That's the secret expensive thing.

### What prompt caching skips on a hit

The cached prefix's KV tensors get **reused as if they were just computed**. The forward pass for those tokens is **completely skipped**:
- 80 layers × N tokens × full attention computation → skipped
- QKV projection matmuls for cached tokens → skipped
- MLP forward passes for cached tokens → skipped

That's the dominant cost of inference. **It's why the discount is 90%, not 5%.**

### Why it's not 100% off

The server still does real work on a cache hit:

| Cost | Description |
|---|---|
| **KV memory residency** | Holding GBs of cache in fast memory isn't free. Recovered through write premium + read rate. |
| **Cache lookup** | Hashing prefix, integrity checks, eviction logic, multi-tenant isolation. |
| **KV transfer** | If the cache lives below GPU HBM, it has to stream onto GPU before decode starts. |
| **Decode-time attention over cached K/V** | Every output token still does an O(N) read across the cached state. Cached prefill is free; cached decode-attention is not. |

10% (the cache read rate) covers all of that.

### Why output isn't discounted

Output tokens go through the **decode** phase, which:
- Was never cacheable in the first place (it's pure generation)
- Is memory-bandwidth-bound, not compute-bound
- Each output token still does one full forward pass through all layers

So output stays at 1× pricing. Caching helps prompts, not generations. **A verbose agent doesn't get cheaper from caching — only the input ingestion does.**

### Why there's a write premium (1.25× for 5-min TTL)

When you mark something cacheable, Anthropic now has to **invest GPU memory on your behalf** for 5 minutes (or 1 hour for the 2× tier). The premium covers:
- Provisioning that memory
- Lookup overhead on every subsequent request
- Eviction risk (your write may displace another customer's hot cache entry)

It's the price of asking for memory residency.

---

## Practical caching rules

### The four canonical breakpoints

You can mark up to **4 cache breakpoints** per request. Place them at the boundaries between "stable" and "changing" content, in order from prompt start:

```
┌──────────────────────────┐
│ System prompt            │ ← breakpoint 1: rarely changes
├──────────────────────────┤
│ Tool definitions (JSON)  │ ← breakpoint 2: changes when tools added
├──────────────────────────┤
│ Long static context      │ ← breakpoint 3: e.g., RAG snippets per session
├──────────────────────────┤
│ Conversation history     │ ← breakpoint 4: grows turn by turn
├──────────────────────────┤
│ Latest user message      │ ← never cached (always changes)
└──────────────────────────┘
```

**Critical rule:** the cache key is the *entire prefix up to the marker*. Change the system prompt and every breakpoint after it is invalidated. Order matters — most stable thing first.

### Three rules that bite

1. **Caches are byte-exact.** A trailing space, a different timestamp embedded in the system prompt, a re-ordered tool list — all cache misses. Linters that "tidy" prompts will silently nuke your hit rate.
2. **Cache is per-API-key + per-org + per-region.** Two app instances on different keys don't share a cache.
3. **Minimum cacheable size:** ~1024 tokens for Sonnet, ~2048 for Opus. Caching tiny prompts costs more than it saves. (We hit this exact issue when our first system prompt was just under 1024 tokens — caching silently did nothing until we extended it.)

### TTLs

| TTL | Write cost | Use when |
|---|---|---|
| **5 minutes** (default) | 1.25× | Active conversations — back-to-back messages |
| **1 hour** | 2× | Re-using big static context across users (e.g., 50k-token system + RAG corpus shared by all sessions) |

The cache is **silently extended** every time you hit it. If user A hits at minute 4, the TTL resets — user B hitting at minute 8 still gets a hit. So 5-minute caches usually outlive their stated TTL in any active app.

### When to NOT cache

- Single-turn one-shot prompts (write premium > savings)
- Prompts where every byte is dynamic
- Prompts under ~1k tokens (below cacheable minimum)
- Prototypes where the prompt changes every iteration

---

## Real numbers from `agent_lg_cached.py`

We ran the same 2-turn agent three times with a ~2000-token system prompt:

```
scenario             fresh    out   c.read   c.create   cost USD
baseline              4468    141        0          0   $0.015519   ← no cache_control
cached cold              4    141     4389         75   $0.003725   ← 76% cheaper
cached warm              4    141     4389         75   $0.003725   ← 76% cheaper
```

### What the numbers say

- **Baseline:** all 4468 input tokens billed at full $3/M = $0.0134 input cost.
- **With caching:** only **4 tokens** at full rate. The other 4389 hit cache at $0.30/M; 75 newly cached at $3.75/M. Input cost drops from $0.0134 to ~$0.0017.
- **76% cheaper per run.** And this is a tiny 2-turn agent. On a 10-turn agent the savings keep compounding.

### "Cold and warm look identical?"

Yes — and that's the most realistic part of the demo. The Anthropic cache is **shared across requests with the same key** within its TTL. Our test runs were close together, so by the time the "cold" run started, an *earlier* run had already populated the cache.

In production this is a **feature**: if user A and user B send messages within 5 minutes that share the same system prompt, **user B benefits from the work user A paid for**. Cache savings amortize across your whole fleet.

### `cache_create=75` on call 2 — incremental extension

On call 1, cache_create was 0 and cache_read was 4389. Then on call 2, cache_create flipped to 75. Why?

Because call 2's prefix is *longer than call 1's*. Call 1 cached the system+tools+user portion (4389 tokens, hit). The 75 new tokens of "AI tool-call message + tool results" got freshly cached on call 2 — meaning a hypothetical call 3 with the same prefix would hit those too.

This is **incremental cache extension**: each turn adds a small new tail to the cache. Multi-turn agents naturally develop deeper cached prefixes turn by turn.

### The 4 "fresh" tokens

Why isn't fresh = 0? A few tokens always change per request — unique parts of the user message that aren't part of the cached prefix, plus tiny per-request metadata. Negligible.

---

## Mental models cheat sheet

The four lines that summarize this whole document:

```
1. The LLM never calls anything. The LLM Client does.
2. The client always sends the full prompt. Every byte. Every time.
3. Caching is a server-side optimization, transparent to the client.
4. The cache discount is real because the server skips prefill — the
   most expensive 80% of the work — not because Anthropic is being nice.
```

If those four sentences make sense in isolation, you have the mental model. Everything else is implementation detail.

---

## Where to go next

You now understand:
- The agent loop (manual + framework versions)
- The token economics of multi-turn conversations
- Who actually executes tool calls (the LLM Client)
- Prompt caching mechanics, pricing, and how to wire it up

Highest-leverage next steps:

1. **Add a checkpointer** — `MemorySaver()` in langgraph gives the agent memory across `.invoke()` calls. Build a chatbot that remembers earlier turns. Watch the cache hit rate climb.
2. **RAG (retrieval-augmented generation)** — load docs → chunk → embed → vector store → retrieve → stuff into prompt. The single most-used LangChain pattern in production. Pairs perfectly with caching: the retrieved context becomes the cacheable prefix.
3. **Structured output** — `model.with_structured_output(PydanticModel)` makes Claude return validated objects instead of strings. The thing that turns LLMs from "demo" into "production."
4. **Real tools** — replace `add` / `get_current_time` with web search (Tavily), SQL, file I/O. See the agent reason over real data.
5. **MCP** — connect your LLM Client to standardized tool servers. Same propose → execute → feedback dance, just over JSON-RPC.

---

## Quick reference — the four files together

```python
# 1. hello.py — the model wrapper
model.invoke("...")

# 2. chain.py — LCEL composition
chain = prompt | model | parser
chain.invoke({"variable": "..."})

# 3. agent.py / agent_lg.py — the agent loop
agent = create_react_agent(model, tools=[...])
agent.invoke({"messages": [("user", "...")]})

# 4. agent_lg_cached.py — caching the stable prefix
cached_system = SystemMessage(content=[
    {"type": "text", "text": LONG_SYSTEM, "cache_control": {"type": "ephemeral"}},
])
agent = create_react_agent(model, tools=[...], prompt=cached_system)
agent.invoke({"messages": [("user", "...")]})
```

Four patterns. One framework. The whole journey from "send a string to an LLM" to "production-grade agent with token caching" in roughly 200 lines of Python total.
