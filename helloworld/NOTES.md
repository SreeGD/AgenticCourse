# LangChain from Zero — A Step-by-Step Walkthrough

A newbie-friendly tour of LangChain, built one file at a time. Each step adds exactly one new concept on top of the previous one. By the end you'll understand the three foundational layers that everything else in LangChain is built on.

**Files in this folder:**

| File | Concept | LLM calls per run |
|---|---|---|
| `hello.py` | Model wrapper — the lowest level | 1 |
| `chain.py` | LCEL composition — `prompt \| model \| parser` | 1 |
| `parallel.py` | LCEL fan-out — run several chains concurrently | N (in parallel) |
| `agent.py` | Tool-calling loop — model decides what to do | N (≥ 2) |

---

## What is LangChain?

**LangChain** is a Python/JS framework for building applications powered by Large Language Models. Its core idea is **composition**: chain together reusable building blocks to go from "send a prompt to an LLM" to full agentic systems.

The main building blocks:

| Block | Purpose |
|---|---|
| **Models** | Wrappers around LLMs (OpenAI, Anthropic, Ollama, etc.) — uniform interface |
| **Prompts** | Templates with variables (`PromptTemplate`, `ChatPromptTemplate`) |
| **Output Parsers** | Convert LLM string output into structured data |
| **Chains (LCEL)** | Compose the above with `\|` — `prompt \| model \| parser` |
| **Retrievers / Vector Stores** | RAG — fetch relevant docs to ground answers |
| **Tools & Agents** | Let the LLM call functions, search, run code |
| **Memory** | Persist conversation state across turns |

The modern way to build is **LCEL** (LangChain Expression Language) — composing runnables with the `|` pipe operator, similar to Unix pipes.

---

## Setup (do this once)

### 1. Create a virtual environment and install dependencies

```bash
cd helloworld
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get an Anthropic API key

Sign up / log in at https://console.anthropic.com/ and create a key. It looks like `sk-ant-api03-...`.

### 3. Configure your key

```bash
cp .env.example .env
# open .env in an editor and replace the placeholder with your real key
```

**File hygiene** — already set up for you in `.gitignore`:

| File | Goes in git? | Contains |
|---|---|---|
| `.env` | ❌ never | your real secret key |
| `.env.example` | ✅ yes | placeholder template |

If you ever paste a real key into `.env.example` by accident, **rotate it** at console.anthropic.com immediately.

---

## Step 1 — `hello.py`: the model wrapper

The simplest possible LangChain program: send one prompt, print one response.

```python
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-6")
response = model.invoke("Say hello in one short sentence.")
print(response.content)
```

**Run it:**
```bash
python hello.py
```

**Expected output (will vary):**
```
Hello! Hope you're having a wonderful day! 😊
```

### What's actually happening

1. `load_dotenv()` reads `.env` and puts `ANTHROPIC_API_KEY` into `os.environ`.
2. `ChatAnthropic(model=...)` is LangChain's wrapper around Anthropic's API. It auto-picks up the API key from the environment.
3. `model.invoke("...")` sends the request and returns an `AIMessage` object (not a string).
4. `response.content` is the actual text. `response` itself also contains token usage, model id, stop reason, etc. — try `print(response)` to see.

### The big idea

The same `.invoke()` interface works for **every** LLM provider:

```python
from langchain_openai import ChatOpenAI         # ChatOpenAI(model="gpt-4o")
from langchain_ollama import ChatOllama         # ChatOllama(model="llama3.2")
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini
```

Swap the import + the constructor and the rest of your code is unchanged. **That's the model abstraction.**

### Try this
- Change the prompt to `"Explain LangChain in 2 sentences."`
- Print the full `response` object instead of `response.content` — see what else is in there
- Swap to `claude-opus-4-7` for the most capable Claude model (more expensive)

---

## Step 2 — `chain.py`: LCEL composition

Now we add **prompt templates** and **output parsers**, and compose them into a pipeline.

```python
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical explainer for senior engineers."),
    ("human", "Explain {topic} like I'm a senior backend engineer, in 3 bullet points."),
])

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "LangChain"})
print(result)
```

**Run it:**
```bash
python chain.py
```

### The pipeline, stage by stage

| Stage | Input | Output |
|---|---|---|
| `prompt` | `{"topic": "LangChain"}` | a list of messages with `{topic}` filled in |
| `model` | the messages | an `AIMessage` from Claude |
| `parser` | the `AIMessage` | the plain `.content` string |

The `|` operator wires them together. The output of each step becomes the input of the next — exactly like Unix pipes.

### The prompt vocabulary (you'll see these terms everywhere)

Four words that are often used interchangeably but mean different things. Get these right early — they show up in every LangChain doc, every blog post, and every Anthropic/OpenAI API reference.

#### 1. Prompt

The **generic** term for the input you send to an LLM. Could be a single string (`"Hello!"`) or a list of role-tagged messages (`[{"role": "user", "content": "Hello!"}]`). When someone says "prompt" without qualification, they usually mean *the whole input bundle going to the model*.

```python
# Both of these are "prompts":
model.invoke("Hello!")                                                  # string prompt
model.invoke([("system", "You are X"), ("human", "Hello!")])           # message-list prompt
```

#### 2. System Prompt

A **special message** that sets the model's persona, rules, style, and constraints **before** the user's first message. The model treats it as background instructions, not as part of the conversation. Modern chat models accept exactly one system prompt per conversation, at the start.

```python
("system", "You are a concise technical explainer for senior engineers.")
                  ↑ a system prompt
```

What you put in a system prompt:
- Persona: *"You are a senior backend engineer."*
- Style rules: *"Respond in bullet points. Avoid emoji."*
- Domain context: *"The user is debugging a PostgreSQL deadlock."*
- Constraints: *"Never invent function names. Cite tools you used."*
- Tool-use guidance: *"Call `search` for any factual lookup."*

The system prompt is also typically the **most cacheable** part of a conversation — it's stable across turns and across users. (See `LEARNINGS.md` for prompt caching.)

#### 3. Human Prompt (a.k.a. User Prompt)

The **user's actual question or input.** It's what the user types into the chatbox — or what your application constructs on the user's behalf. The model treats this as the thing it should respond to.

```python
("human", "Explain LangChain in 3 bullet points.")
                  ↑ a human prompt
```

In a multi-turn conversation, you'll send multiple human prompts (one per user turn), interleaved with the model's `AIMessage` responses.

| LangChain class | Role string | What it represents |
|---|---|---|
| `SystemMessage` | `"system"` | the system prompt |
| `HumanMessage` | `"human"` or `"user"` | a human prompt |
| `AIMessage` | `"ai"` or `"assistant"` | the model's response |
| `ToolMessage` | `"tool"` | a tool's return value (see `agent.py`) |

#### 4. Prompt Template

A **reusable string with `{variable}` placeholders** that you fill in at runtime. Templates separate the *structure* of a prompt from the *data* that varies per request — the same idea as parameterized SQL or Python's f-strings, but built for LangChain's pipeline interface.

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Explain {topic} in {n} bullet points."
)

template.invoke({"topic": "LangChain", "n": 3})
# → "Explain LangChain in 3 bullet points."
```

`PromptTemplate` produces a **single string**. Useful for completion-style models or when you want one big text blob.

#### 5. Human Prompt Template / System Prompt Template

Templates **specifically for one role** within a chat-style conversation. Each one fills its own variables, then they get composed together by `ChatPromptTemplate`.

```python
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_tmpl = SystemMessagePromptTemplate.from_template(
    "You are a {persona} for {audience}."
)

human_tmpl = HumanMessagePromptTemplate.from_template(
    "Explain {topic} in {n} bullet points."
)
```

Each template knows its **role** (`system` / `human`) and produces a typed message (`SystemMessage` / `HumanMessage`) when invoked, not a raw string.

#### 6. ChatPromptTemplate (puts it all together)

The class that **composes multiple role-specific templates** into a full chat prompt. This is what we use in `chain.py`. It accepts a list of `(role, template_string)` tuples (LangChain auto-creates the role-specific template under the hood) or pre-built templates.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {persona} for {audience}."),       # ← system prompt template
    ("human",  "Explain {topic} in {n} bullet points."),     # ← human prompt template
])

prompt.invoke({
    "persona": "concise technical explainer",
    "audience": "senior engineers",
    "topic": "LangChain",
    "n": 3,
})
# → [SystemMessage("You are a concise technical explainer for senior engineers."),
#    HumanMessage("Explain LangChain in 3 bullet points.")]
```

Output of `ChatPromptTemplate` = a **list of messages** ready for `model.invoke()`.

#### Quick visual

```
┌─────────────────────────────────────────────────────────────────┐
│                     ChatPromptTemplate                          │
│  ┌────────────────────────────┐  ┌───────────────────────────┐  │
│  │ SystemMessagePromptTemplate│  │ HumanMessagePromptTemplate│  │
│  │  "You are a {persona}..."  │  │  "Explain {topic}..."     │  │
│  └────────────┬───────────────┘  └────────────┬──────────────┘  │
│               │ .invoke({...})                │ .invoke({...})  │
│               ▼                                ▼                │
│  ┌────────────────────────────┐  ┌───────────────────────────┐  │
│  │      SystemMessage         │  │       HumanMessage        │  │
│  │   ("system prompt")        │  │     ("human prompt")      │  │
│  └────────────────────────────┘  └───────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
                   [SystemMessage, HumanMessage]
                            (a "prompt")
                                   │
                                   ▼
                            model.invoke(...)
```

#### Mental model in one line

> **A "prompt" is the input to the model. A "prompt template" is a recipe for building a prompt. The role of a message (system / human / ai / tool) tells the model how to interpret it.**

### Why this matters: LCEL

Every LangChain component implements a `Runnable` interface, which means **the same chain object** automatically supports:

```python
chain.invoke({"topic": "RAG"})                   # synchronous, single
chain.batch([{"topic": "RAG"}, {"topic": "MCP"}]) # parallel batch
async for chunk in chain.astream({"topic": "agents"}):  # streaming
    print(chunk, end="", flush=True)
```

You write the chain *once*, and you get sync / async / streaming / batch behavior for free.

### A note on `temperature`

`temperature=0` makes the model produce **the same answer for the same input** (almost — see below). Default is around `0.7`, which adds creative variation.

| Setting | When to use |
|---|---|
| `temperature=0` | Tutorials, tests, structured extraction, evaluation harnesses |
| `temperature=0.7` | Creative writing, brainstorming, chat |
| `temperature=1+` | Pushing for novelty (rarely useful) |

**Caveat:** even at `temperature=0`, LLM outputs aren't bit-exact across runs. GPU non-determinism and tie-breaking introduce small variation. Treat `temperature=0` as **"strongly deterministic,"** not perfectly deterministic.

### Try this
- Change `{"topic": "LangChain"}` to a different topic — same chain, different question
- Add a second variable like `{tone}` to the human message and pass `{"topic": "...", "tone": "casual"}`
- Replace `StrOutputParser` with `JsonOutputParser` and ask Claude to respond in JSON

---

## Step 3 — `agent.py`: tools and the agent loop

The biggest jump: now the LLM can request that **functions on your machine** be called, get the results back, and produce an answer using them.

We define two tools — `add` and `get_current_time` — and build the agent loop manually so you can see exactly what's happening.

```python
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat(timespec="seconds")


tools = [add, get_current_time]
tools_by_name = {t.name: t for t in tools}

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools(tools)

history = [HumanMessage("What's 47 plus 158, and what time is it right now? Tell me both.")]

turn = 1
while True:
    print(f"\n--- turn {turn}: calling model ---")
    ai_msg = model.invoke(history)
    history.append(ai_msg)

    if not ai_msg.tool_calls:
        print("\n--- final answer ---")
        print(ai_msg.content)
        break

    for tc in ai_msg.tool_calls:
        print(f"  tool_call: {tc['name']}({tc['args']})")
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        print(f"  -> {result}")
        history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    turn += 1
```

**Run it:**
```bash
python agent.py
```

**Expected trace:**
```
--- turn 1: calling model ---
  tool_call: add({'a': 47, 'b': 158})
  -> 205
  tool_call: get_current_time({})
  -> 2026-05-09T22:07:38

--- turn 2: calling model ---

--- final answer ---
1. **47 + 158 = 205**
2. **The current local time is 10:07 PM on May 9, 2026.**
```

### Reading the trace

**Turn 1** — Claude saw the user's question + the auto-generated schemas of your two tools (built from your `@tool` docstrings + type hints). Instead of producing prose, Claude returned an `AIMessage` with **two `tool_calls`** (and empty `.content`). Your loop ran both tools and appended the results as `ToolMessage`s.

**Turn 2** — Claude saw the full history including the tool results. It now has all the information it needs and produces the natural-language answer. No more `tool_calls` → loop exits.

### The most important concept in this whole tutorial

> **The LLM never calls anything. The LLM Client does.**

Read that twice. It's *the* mental model that makes agents stop feeling magical.

The LLM is a pure text function — text in, text out. It cannot execute code, hit a clock, open a network connection, or read a file. Ever.

What it *can* do is **emit JSON saying "I'd like `add` called with `{a: 47, b: 158}`."** Your code — the **LLM Client** — is what:

1. Decides whether to honor that request
2. Looks up the function
3. Runs it on your CPU
4. Captures the result
5. Sends the result back to the LLM as more text on the next turn

```
  +--------+                       +-----------------+
  |  YOU   |                       |  Anthropic API  |
  |  (LLM  |                       |    (Claude)     |
  | Client)|                       |                 |
  +---+----+                       +--------+--------+
      |                                     |
      |  prompt + tool schemas              |
      |------------------------------------>|
      |                                     |
      |     "please call add(47,158)"       |
      |<------------------------------------|
      |                                     |
      |  *** YOU run add(47,158) → 205 ***  |
      |       (Claude has no idea           |
      |        this is happening)           |
      |                                     |
      |  prompt + history + result=205      |
      |------------------------------------>|
      |                                     |
      |     "47+158 is 205. ..."            |
      |<------------------------------------|
      v                                     v
```

### Why this design wins

- **Security** — the model can't `rm -rf /` because it can't run anything. It can only ask. You decide whether to honor the request.
- **Auditability** — every action is your Python code; you can log, rate-limit, or reject any call.
- **Portability** — a tool can be a Python function, a SQL query, a REST call, a Bash command, an MCP server, a robot arm. The LLM doesn't know or care.
- **Determinism where it matters** — your `add` function is deterministic Python. Only the *decision* to call it is fuzzy LLM output.

### Vocabulary, decoded

| Term | What it really is |
|---|---|
| **Agent** | An LLM Client that runs the propose → execute → feedback loop |
| **Tool** | A function the LLM Client is willing to run on the LLM's behalf |
| **MCP server** | A remote toolbox; the LLM Client connects to it to discover and invoke tools |
| **MCP client / host** | The LLM Client, but with the tool layer abstracted over a network protocol |
| **Multi-agent system** | Multiple LLM Clients (or one Client orchestrating multiple LLM personas) |
| **Guardrails / approvals** | Rules the LLM Client enforces *before* honoring a tool request |

### Try this

1. **Break a tool intentionally** — `raise ValueError("nope")` inside `add`. Watch Claude get the error as a `ToolMessage` and recover.
2. **Give it an unanswerable question** — `"What's the weather in Mumbai?"` with no weather tool. Claude will admit it has no tool for that instead of hallucinating.
3. **Add a sequential dependency** — define `multiply(a, b)` and ask `"What is (47+158) times 3?"`. You'll see **3 turns**: Claude calls `add`, *waits for the result*, then calls `multiply(a=205, b=3)`. That's the moment "tool calling" becomes "agent reasoning."

---

## Step 4 — `parallel.py`: parallel chains (LCEL fan-out)

So far each chain has been a straight line: `prompt | model | parser`. Now we **branch out**: run several chains *at the same time* against the same input, then collect their results into one dict.

This is the second LCEL primitive (after the `|` pipe): **`RunnableParallel`**.

### The shape

```
                    ┌─► eli5_chain    ── "Explain like I'm 5"
input: {topic: X}  ─┼─► senior_chain  ── "Explain to a senior engineer"
                    └─► haiku_chain   ── "Write a haiku"
                                                    │
                                                    ▼
                {"eli5": "...", "senior": "...", "haiku": "..."}
```

Three branches, one input dict, one merged output dict. Each branch is a normal LCEL chain — they're not aware they're running in parallel.

### Code (the heart of `parallel.py`)

```python
from langchain_core.runnables import RunnableParallel

def make_chain(template):
    prompt = ChatPromptTemplate.from_messages([("human", template)])
    return prompt | model | parser

eli5_chain   = make_chain("Explain {topic} like I'm 5...")
senior_chain = make_chain("Explain {topic} to a senior engineer...")
haiku_chain  = make_chain("Write a haiku about {topic}...")

parallel = RunnableParallel(
    eli5=eli5_chain,
    senior=senior_chain,
    haiku=haiku_chain,
)

result = parallel.invoke({"topic": "prompt caching"})
# {"eli5": "...", "senior": "...", "haiku": "..."}
```

### Run it

```bash
python parallel.py
```

The file times both a sequential run (each chain in turn) and a parallel run (all at once), so you can see the speedup directly.

### Real numbers from a sample run

```
SEQUENTIAL:
  eli5:    3.94s
  senior:  6.40s
  haiku:   1.29s
  TOTAL:  11.63s        ← sum of branches

PARALLEL:
  TOTAL:   6.59s        ← max of branches
  Speedup: 1.77×
```

**Wall-clock time = the slowest branch**, not the sum. That's the whole point.

### Why is the speedup 1.77× and not 3×?

Three reasons, in order of impact:

1. **Branches finish at different times.** The haiku branch finished in 1.3s and then sat idle waiting for `senior` to finish. Parallelism is bounded by your *slowest* branch.
2. **Coordination overhead.** `RunnableParallel` wraps each branch in a thread (sync mode) or task (async mode). Small fixed cost.
3. **Server-side variance.** Three concurrent requests to Anthropic land on different GPU pods; under load some serialize.

To get closer to 3×: make the branches **similar in size**, run them **async**, and use a model with consistent latency.

### Two equivalent forms

`RunnableParallel(eli5=..., senior=...)` and the dict-literal `{"eli5": ..., "senior": ...}` produce the same thing. LCEL automatically wraps a plain dict into a `RunnableParallel` when it's piped into the next runnable:

```python
# These two are equivalent:
parallel = RunnableParallel(eli5=eli5_chain, senior=senior_chain, haiku=haiku_chain)

parallel = {"eli5": eli5_chain, "senior": senior_chain, "haiku": haiku_chain}
chain = parallel | next_step   # dict gets auto-promoted to RunnableParallel here
```

The dict-literal form is the idiomatic one in production code. Use whichever reads more clearly.

### The async path is faster

LLM calls are I/O-bound (waiting for the network), not CPU-bound. For I/O-bound parallelism, **async beats threads** — no thread-pool overhead, just `asyncio.gather()`:

```python
result = await parallel.ainvoke({"topic": "prompt caching"})
```

If you're inside FastAPI, Quart, or any async runtime, use `.ainvoke()`. The sync `.invoke()` runs branches in a `concurrent.futures.ThreadPoolExecutor` — it works, but it's heavier.

### Where this pattern shines in real apps

| Use case | What the branches do |
|---|---|
| **Multi-aspect analysis** | Classify, summarize, extract entities, score sentiment — all on the same document |
| **Multi-language translation** | One branch per target language, single source text |
| **Retrieval ensembles** | Query 3 vector stores or retrievers in parallel, merge results |
| **Map-reduce summarization** | Summarize each chunk in parallel, then a final reducer chain |
| **A/B prompt testing** | Same input, two prompts, compare outputs side by side |
| **Multi-model voting** | Same prompt to Claude + GPT + Gemini, take majority answer |

The last one is especially powerful with LangChain's model abstraction — swap the model in each branch and you've got cross-provider ensembling in five lines.

### What `RunnableParallel` does NOT do

- **It doesn't share state between branches.** Each branch sees the input independently. If branch B needs branch A's output, that's *sequential* — pipe `parallel_step | next_step` so the next step receives the merged dict.
- **It doesn't deduplicate work.** If two branches send the same prompt, they make two API calls. Use a `RunnableLambda` upstream to compute shared values once and pass them down.
- **It doesn't bound parallelism.** Three branches → three concurrent requests. Ten branches → ten. Add an external rate limiter if your API key has tight RPM limits.

### Try this

1. **Add a 4th branch** — e.g. `tweet_chain` ("write a 280-char tweet about {topic}"). Watch the parallel time stay roughly the same; sequential time grows.
2. **Swap to async** — change `.invoke()` to `await .ainvoke()` (you'll need to wrap the script in `async def main()` and run with `asyncio.run`). Compare wall-clock.
3. **Add a synthesizer step** — after the parallel fan-out, pipe the merged dict into a final chain that combines the three perspectives into one answer:
   ```python
   synthesizer = (
       ChatPromptTemplate.from_messages([
           ("human", "ELI5: {eli5}\n\nSenior: {senior}\n\nHaiku: {haiku}\n\nWrite a one-paragraph synthesis."),
       ])
       | model
       | parser
   )
   chain = parallel | synthesizer
   ```
   This is the classic **map-reduce** shape: parallel branches (map) → single combiner (reduce).

### Mental model in one line

> **`prompt | model | parser` is a sequential pipe (one stage feeds the next). `{"a": chain_a, "b": chain_b}` is a parallel fan-out (one input, many simultaneous chains, merged output). Together they cover almost every LCEL pattern you'll write.**

---

## Where to go next

You've now seen the three foundational layers. Everything else in LangChain (RAG, multi-agent, memory, structured output) is built on top of them.

Suggested next steps, in order of value:

- **`create_react_agent` from `langgraph`** — the production version of the manual loop in `agent.py`. ~5 lines, plus you get streaming/checkpointing/human-in-the-loop for free. Same dance, productionized.
- **A real tool** — wire up a web search tool (e.g. Tavily, DuckDuckGo) or a SQL tool. See how the model reasons over fresh / real-world data.
- **Structured output** — `model.with_structured_output(MyPydanticModel)` makes Claude return validated objects instead of strings.
- **RAG** — load documents → chunk → embed → store in a vector DB → retrieve relevant chunks at query time → stuff them into the prompt. Same LCEL, with a `retriever` step in the chain.
- **MCP** — connect your LLM Client to *any* MCP server and expose its tools to Claude. Same propose → execute → feedback dance, just over a standardized protocol.

---

## Quick reference — the four patterns

```python
# 1. Just call the model
model.invoke("...")

# 2. Compose a chain (sequential pipe)
chain = prompt | model | parser
chain.invoke({"variable": "..."})

# 3. Fan out across chains (parallel)
parallel = RunnableParallel(a=chain_a, b=chain_b, c=chain_c)
# OR equivalently:  parallel = {"a": chain_a, "b": chain_b, "c": chain_c}
parallel.invoke({"shared_input": "..."})
# → {"a": ..., "b": ..., "c": ...}

# 4. Loop until the model stops requesting tools
model_with_tools = model.bind_tools([tool_a, tool_b])
while True:
    msg = model_with_tools.invoke(history)
    history.append(msg)
    if not msg.tool_calls:
        break
    for tc in msg.tool_calls:
        result = registry[tc["name"]].invoke(tc["args"])
        history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
```

Four patterns. That's the whole foundation. Everything else is variations on these.
