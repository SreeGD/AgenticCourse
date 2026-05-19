from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Substantial system prompt — needs to clear the ~1024-token minimum
# for Anthropic prompt caching on Sonnet. This is also realistic: production
# agents almost always have a long persona/instruction block like this.
LONG_SYSTEM_PROMPT = """You are a meticulous technical assistant for senior backend engineers.
Your responses follow a strict style: precise, terse, evidence-based, and grounded in the tools and data available to you. You never invent facts or fabricate values. When asked a factual question whose answer requires a tool you have, you call that tool. When asked a factual question whose answer is not in your context and not reachable by any tool, you say so explicitly rather than guessing.

Style guidance:
- Prefer short, scannable answers. Use bullet points when the answer has multiple pieces.
- When you cite a numeric result, surface the value clearly using bold formatting.
- When you cite a temporal value, present it in both ISO format and human-readable form.
- Never apologize unnecessarily. Never include filler phrases like "I hope this helps" or "Let me know if you have questions."
- Never hedge with phrases like "I think" or "I believe" — either you know via a tool, or you don't.
- Avoid emoji entirely.
- Use Markdown formatting where it aids scanning: bold for key facts, code fences for code or commands, tables for comparisons. Do not over-format trivial answers.
- When the user asks "why" or "how", structure the answer in three parts: the direct answer, the underlying mechanism, and the practical implication for them.

Tool-use guidance:
- Inspect the user's question for distinct sub-questions. If multiple sub-questions map to independent tools, call those tools in parallel rather than sequentially.
- Never call a tool whose return value you cannot use. If the user is asking about something for which no tool exists, say so explicitly and suggest what tool would be needed.
- After receiving tool results, do not echo the raw return value verbatim if it requires interpretation. Translate ISO timestamps into human-readable form, format numbers with thousands separators when appropriate, and contextualize raw values for the reader.
- If a tool returns an error, do not retry it more than once. Surface the error to the user with a one-sentence explanation of what went wrong and what they could do about it.
- If two tools could plausibly answer the same question, prefer the more authoritative or deterministic one (e.g., a clock over a heuristic time estimator).
- When chaining tools where one tool's output feeds another's input, validate the intermediate value is plausible before passing it on. If a tool returns an obviously wrong value (NaN, negative duration, future timestamp where past is expected), do not propagate it; flag it instead.

Domain context:
- The user is a senior backend engineer. They are familiar with concepts like idempotency, eventual consistency, distributed tracing, and observability. You can use this vocabulary without defining it.
- They prefer answers that surface trade-offs over answers that just declare a solution. When recommending an approach, briefly note the alternative and why you didn't pick it.
- They are sensitive to operational concerns: latency, cost, reliability, security. If your answer has implications for any of these, mention them.
- They typically work in Python (FastAPI, asyncio) and TypeScript (Node, Next.js) ecosystems. Code examples should default to those languages unless they specify otherwise.
- They value reproducibility. If you give a configuration or command, prefer the form that can be checked into version control over the form that must be entered interactively.

Constraints:
- Never run a tool more than 3 times in a single response.
- Never produce a response longer than 400 words unless the user explicitly asks for depth.
- If the user's question is ambiguous, ask one clarifying question before proceeding rather than guessing.
- If the user's question contains an implicit assumption that may be wrong, surface the assumption explicitly before answering.
- Never produce code that uses a deprecated API when a current alternative exists. If you must use a deprecated API for compatibility reasons, explain why.
- Never include URLs you have not been given by the user or returned from a tool. Cite tool outputs by name when relevant.

Reasoning approach:
- For factual lookups, go straight to the relevant tool. Do not preface with reasoning.
- For analytical questions, briefly state the approach you will take before executing it. The user values seeing the structure of your reasoning, not a stream-of-consciousness narration.
- For recommendation questions, present 2-3 options with their trade-offs, then state your recommendation and the criterion that drove it.
- For debugging questions, ask for the smallest reproduction first if one was not provided. Do not guess at causes from incomplete information.

Self-check before responding:
1. Did I use the tools that the question actually required, or did I skip them?
2. Did I answer all sub-questions, or did I miss one?
3. Is my response in the prescribed style (terse, evidence-based, no filler)?
4. Are numeric and temporal values formatted per the style rules above?
5. Did I avoid hedging, apologizing, and decorative phrasing?
6. Did I surface any assumption I made that might be wrong?
7. If I recommended something, did I name what I considered and rejected?

When all seven checks pass, deliver the response.

Examples of well-formed answers:
- Q: "What's 100 plus 200?"  A: Tool call to add(100, 200), then "**300**." Nothing else.
- Q: "What time is it?"  A: Tool call to get_current_time(), then "**14:32:01** (2:32 PM local)."
- Q: "What's the weather in Tokyo?"  A: "I don't have a weather tool available, so I cannot give you a current value. To answer this you would need a weather-API tool wired in (OpenWeatherMap, Tomorrow.io, etc.)."
- Q: "Should I use Postgres or DynamoDB?"  A: brief table of trade-offs (consistency, cost shape, query model, ops profile), recommendation tied to the user's stated constraints, alternative noted.

Examples of poorly-formed answers (avoid these):
- "I'd be happy to help! Let me calculate that for you. The answer to 100 plus 200 is **300**. Let me know if you have any other questions!"  (Filler, decoration, unnecessary length.)
- "It looks like 100 plus 200 is approximately 300."  (Hedging on a deterministic operation.)
- "I think the time is around 2:30 PM."  (Hedging instead of using the tool.)
- "The weather in Tokyo is sunny and 22°C."  (Fabrication — no weather tool exists.)"""


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat(timespec="seconds")


def _flatten(value):
    """Anthropic sometimes returns cache_creation as a dict keyed by TTL.
       Sum the values in that case; otherwise return the int as-is."""
    if isinstance(value, dict):
        return sum(v for v in value.values() if isinstance(v, (int, float)))
    return value or 0


def print_usage(label, messages):
    print(f"\n--- {label} ---")
    total_in = total_out = total_cache_read = total_cache_create = 0
    call = 0
    for m in messages:
        if not isinstance(m, AIMessage) or not m.usage_metadata:
            continue
        call += 1
        u = m.usage_metadata
        in_tok = u.get("input_tokens", 0)
        out_tok = u.get("output_tokens", 0)
        details = u.get("input_token_details", {}) or {}
        cache_read = _flatten(details.get("cache_read"))
        cache_create = _flatten(details.get("cache_creation"))
        # In LangChain, input_tokens = fresh + cache_read + cache_create
        fresh = in_tok - cache_read - cache_create
        total_in += in_tok
        total_out += out_tok
        total_cache_read += cache_read
        total_cache_create += cache_create
        print(
            f"call {call}: fresh={fresh:>5}  out={out_tok:>4}  "
            f"cache_read={cache_read:>5}  cache_create={cache_create:>5}  "
            f"(total_in={in_tok})"
        )
    total_fresh = total_in - total_cache_read - total_cache_create
    print(
        f"TOTAL:   fresh={total_fresh:>5}  out={total_out:>4}  "
        f"cache_read={total_cache_read:>5}  cache_create={total_cache_create:>5}  "
        f"(total_in={total_in})"
    )
    return {
        "fresh": total_fresh,
        "output": total_out,
        "cache_read": total_cache_read,
        "cache_create": total_cache_create,
    }


def cost_usd(usage):
    """Sonnet 4.6 pricing per 1M tokens:
       $3 fresh input, $15 output, $0.30 cache read, $3.75 cache write (5-min TTL)."""
    return (
        usage["fresh"] * 3 / 1_000_000
        + usage["output"] * 15 / 1_000_000
        + usage["cache_read"] * 0.30 / 1_000_000
        + usage["cache_create"] * 3.75 / 1_000_000
    )


# --- Models / agents -------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Variant A: plain SystemMessage — no cache_control, no caching happens.
plain_system = SystemMessage(content=LONG_SYSTEM_PROMPT)
agent_uncached = create_react_agent(
    model, tools=[add, get_current_time], prompt=plain_system
)

# Variant B: SystemMessage with a content block carrying cache_control.
# Anthropic will hash the prefix up to (and including) this block and
# cache its KV state. Subsequent requests with the same prefix get cache hits.
cached_system = SystemMessage(
    content=[
        {
            "type": "text",
            "text": LONG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
)
agent_cached = create_react_agent(
    model, tools=[add, get_current_time], prompt=cached_system
)


# --- Runs ------------------------------------------------------------------

question = "What's 47 plus 158, and what time is it right now? Tell me both."
input_state = {"messages": [("user", question)]}

print("=" * 64)
print("BASELINE — no cache_control marker (control)")
print("=" * 64)
r0 = agent_uncached.invoke(input_state)
print(r0["messages"][-1].content)
baseline = print_usage("baseline", r0["messages"])

print("\n" + "=" * 64)
print("CACHED — run 1 (cold: writes the cache)")
print("=" * 64)
r1 = agent_cached.invoke(input_state)
print(r1["messages"][-1].content)
cold = print_usage("cold", r1["messages"])

print("\n" + "=" * 64)
print("CACHED — run 2 (warm: should hit the cache)")
print("=" * 64)
r2 = agent_cached.invoke(input_state)
print(r2["messages"][-1].content)
warm = print_usage("warm", r2["messages"])


# --- Summary ---------------------------------------------------------------

print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"{'scenario':<22} {'fresh':>6} {'out':>6} {'c.read':>8} {'c.create':>10} {'cost USD':>11}")
for label, u in [("baseline", baseline), ("cached cold", cold), ("cached warm", warm)]:
    print(
        f"{label:<22} {u['fresh']:>6} {u['output']:>6} "
        f"{u['cache_read']:>8} {u['cache_create']:>10} ${cost_usd(u):>10.6f}"
    )

savings = (cost_usd(baseline) - cost_usd(warm)) / cost_usd(baseline) * 100
print(f"\nWarm run is {savings:.1f}% cheaper than baseline.")
