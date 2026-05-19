# AgenticCourse Curriculum — Full Plan

**40 sessions • ~80 hours • 13 weeks** (Track M optional; core is 37 sessions / 74 hours / 12 weeks)

Time budget: 1 hour Mon-Fri + 2 hours Sat-Sun = 9 hours/week

The accompanying spreadsheet is `CURRICULUM.csv` — open in Excel / Google Sheets / Numbers.

---

## Phase summary

| Phase | Sessions | Hours | Weeks | Focus |
|---|---|---|---|---|
| **1 — Technical Foundation** | 1-17 | 34 | 1-5 | Every primitive needed for agentic AI |
| **2 — Architect Skills** | 18-21 | 8 | 6 | Judgement, security, governance, product thinking |
| **3 — Vertical Deep Dives** | 22-37 | 32 | 7-12 | Healthcare, Agriculture, Finance, Vidya Karana, Family AI |
| **M — Claude Code Mastery (optional)** | 38-40 | 5 | 13 | Mastering the tool you're already using |

---

## Track breakdown

| Track | Sessions | Count | Theme |
|---|---|---|---|
| A — Agentic Patterns | 1, 2, 3 | 3 | MCP, Reflection/PE, Multi-agent, LTM + Episodic Memory |
| B — Workflow & Skill | 4, 5, 6 | 3 | SDD, Vibe Coding, Claude Skills |
| C — Alt Architectures | 7, 8 | 2 | Anthropic SDK direct, AI Gateway |
| D — Data & Multi-modal | 9 | 1 | Files, Vision, Citations, Batches, multimodal RAG |
| E — Graph Depth | 10 | 1 | Custom LangGraph + HITL |
| E.5 — RAG Architectures | 11, 12, 13 | 3 | Hybrid RAG, GraphRAG, Corrective RAG |
| F — Production | 14, 15, 16, 17 | 4 | Eval, Cost, Streaming, Deploy + Observability |
| G — Architect Skills | 18, 19, 20, 21 | 4 | Interview, Red-teaming, **Governance & Audit (NEW)**, UX |
| H — Healthcare | 22, 23, 24 | 3 | Landscape, CDS Architecture, HIPAA Chatbot |
| I — Agriculture | 25, 26, 27 | 3 | Landscape, Crop Diagnostic, Farmer Bot |
| J — Finance | 28, 29, 30 | 3 | Landscape, Fraud + Support, Investment Research |
| K — Vidya Karana | 31, 32, 33 | 3 | Wellness + Yoga + Applied Vedic Wisdom |
| L — Family AI Agent | 34, 35, 36, 37 | 4 | Multi-generational, multi-channel, multi-specialist |
| **M — Claude Code Mastery (optional)** | 38, 39, 40 | 3 | **CLAUDE.md best practices, Hooks, Autonomous workflows** |

---

## What's new in this revision

Per gap analysis against Brij Kishore Pandey's 9-Concepts + RAG-Architectures + Knowledge-Graph + Claude-Code infographics:

**Newly added (8 sessions total across 3 revisions):**

| Session | Why |
|---|---|
| 08 — AI Gateway | Brij concept 04 — LiteLLM / OpenRouter / Vercel AI Gateway |
| 11 — Hybrid RAG | Brij RAG 01 — dense + BM25 + Reciprocal Rank Fusion |
| 12 — GraphRAG | Brij RAG 02 — entity extraction + knowledge graph |
| 13 — Corrective RAG | Brij RAG 04 — retrieval grader + query rewriter |
| **20 — AI Governance & Audit** | **Brij Knowledge Graph — Confidence scoring + Policy gates + Audit trails + Decision logs** |
| **38 — CLAUDE.md + Settings (Track M)** | **Brij Claude Code Guide — mastering the tool itself** |
| **39 — Claude Code Hooks (Track M)** | **Brij Claude Code Guide — Hooks automation** |
| **40 — Autonomous Workflows (Track M)** | **Brij Claude Code Guide — putting it all together** |

**Scope-expanded:**

| Session | What changed |
|---|---|
| 03 — Multi-agent + LTM | Added **Episodic Memory** as third memory type |
| 09 — Files & Document AI | Added CLIP/ColPali unified multimodal embedding |
| 17 — Production Deployment | Folded in observability |

---

## Calendar (13 weeks, last one optional)

| Week | Sessions | Hours | Theme |
|---|---|---|---|
| 1 | 1, 2, 3 | 6 | Agentic patterns |
| 2 | 4, 5, 6 | 6 | Workflow & skills |
| 3 | 7, 8, 9 | 6 | Alt arch + Data/multi-modal |
| 4 | 10, 11, 12, 13 | 8 | Custom Graph + RAG architectures |
| 5 | 14, 15, 16, 17 | 8 | Production (Eval → Cost → Streaming → Deploy) |
| 6 | 18, 19, **20**, 21 | 8 | Architect skills (incl. **Governance**) |
| 7 | 22, 23, 24 | 6 | **Healthcare** |
| 8 | 25, 26, 27 | 6 | **Agriculture** |
| 9 | 28, 29, 30 | 6 | **Finance** |
| 10 | 31, 32, 33 | 6 | **Vidya Karana** |
| 11 | 34, 35 | 4 | **Family AI** part 1 |
| 12 | 36, 37 | 4 | **Family AI** capstone |
| 13 (optional) | **38, 39, 40** | **5** | **Claude Code Mastery** |

---

## Coverage vs. external benchmarks

### Brij's "9 AI Concepts"

| # | Concept | Where |
|---|---|---|
| 01 | Agentic Loops | Lessons 03, 13 ✓ |
| 02 | MCP | Lesson 12 ✓ |
| 03 | Subagents & Multi-Agent | Lesson 14 ✓ |
| 04 | AI Gateway | Session 8 |
| 05 | Inference Economics | Lesson 04 ✓ |
| 06 | Evals | Session 14 |
| 07 | Guardrails | Lesson 10 ✓ |
| 08 | Observability | Session 17 |
| 09 | The Bitter Lesson | Mindset / research literacy |

### Brij's "Top 5 RAG Architectures"

| # | Architecture | Where |
|---|---|---|
| 01 | Hybrid RAG | Session 11 |
| 02 | GraphRAG | Session 12 |
| 03 | Agentic RAG | Lessons 11, 14 ✓ |
| 04 | Corrective RAG | Session 13 |
| 05 | Multimodal RAG | Session 9 (expanded) |

### Brij's "Agentic AI Knowledge Graph" (governance cluster)

| Concept | Where |
|---|---|
| Confidence Scoring | Session 20 |
| Policy Gates | Session 20 |
| Audit Trail Design | Session 20 |
| Decision Reasoning Logs | Session 20 |
| Episodic Memory | Session 3 (extended) |
| Multi-Modal Reasoning | Session 9 (expanded) |
| Memory & Learning Loops | Frontier — not in curriculum (research stage) |
| Self-Improving Agents | Frontier — not in curriculum (research stage) |
| Agentic Workflows at Scale | Session 17 (deployment scope) |

### Brij's "Claude Code Complete Guide"

| Component | Where |
|---|---|
| CLAUDE.md best practices | **Session 38 (Track M)** |
| Skills | Session 6 |
| Hooks | **Session 39 (Track M)** |
| MCP Servers | Session 1 ✓ |
| Subagents | Session 3 ✓ |
| Autonomous workflows | **Session 40 (Track M)** |

---

## How to use this

**Linear core path:** Sessions 1 → 37 (~12 weeks, 74 hours)

**Optional Track M:** Sessions 38-40 — bonus if you want to master Claude Code itself.

**Reference path:** [`NOTES.md`](./NOTES.md) → [`lessons/`](./lessons/) for per-topic walkthroughs.

---

## File map

```
labs/
├── *.py                    ← runnable examples (one per built lesson)
├── lessons/                ← per-topic markdown walkthroughs (15 done so far)
├── NOTES.md                ← index into lessons/
├── LEARNINGS.md            ← conceptual deep-dives
├── CURRICULUM.md           ← THIS file
├── CURRICULUM.csv          ← spreadsheet tracker (40 rows)
└── requirements.txt
```
