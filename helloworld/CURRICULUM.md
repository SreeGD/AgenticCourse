# AgenticCourse Curriculum — Full Plan

**32 sessions • ~64 hours • 11 weeks**
Time budget: 1 hour Mon-Fri + 2 hours Sat-Sun = 9 hours/week

The accompanying spreadsheet is `CURRICULUM.csv` — open in Excel / Google Sheets / Numbers.

---

## Phase summary

| Phase | Sessions | Hours | Weeks | Focus |
|---|---|---|---|---|
| **1 — Technical Foundation** | 1-13 | 26 | 1-4 | Build every primitive needed for agentic AI |
| **2 — Architect Skills** | 14-16 | 6 | 5 | Judgement, security, product thinking |
| **3 — Vertical Deep Dives** | 17-32 | 32 | 6-11 | Healthcare, Agriculture, Finance, Vidya Karana, Family AI |

---

## Track breakdown

| Track | Sessions | Count | Theme |
|---|---|---|---|
| A — Agentic Patterns | 1, 2, 3 | 3 | MCP, Reflection, Plan-and-Execute, Multi-agent, LTM |
| B — Workflow & Skill Patterns | 4, 5, 6 | 3 | SDD, Vibe Coding, Claude Skills |
| C — Alternative Architectures | 7 | 1 | Anthropic SDK / Claude Agent SDK (no LangChain) |
| D — Data & Multi-modal | 8 | 1 | Files, Vision, Citations, Batches |
| E — Graph Depth | 9 | 1 | Custom LangGraph + HITL |
| F — Production | 10, 11, 12, 13 | 4 | Eval, Cost Optimization, Streaming, Deployment |
| G — Architect Skills | 14, 15, 16 | 3 | System Design Interview, Red-teaming, AI UX |
| H — Healthcare | 17, 18, 19 | 3 | Landscape, CDS Architecture, HIPAA Chatbot |
| I — Agriculture | 20, 21, 22 | 3 | Landscape, Crop Diagnostic, Farmer Bot |
| J — Finance | 23, 24, 25 | 3 | Landscape, Fraud + Support, Investment Research |
| K — Vidya Karana | 26, 27, 28 | 3 | Wellness + Yoga + Applied Vedic Wisdom |
| L — Family AI Agent | 29, 30, 31, 32 | 4 | Multi-generational, multi-channel, multi-specialist |

---

## Calendar (11 weeks)

| Week | Sessions | Hours | Theme |
|---|---|---|---|
| 1 | 1, 2, 3 | 6 | Agentic patterns (MCP → Reflection/PE → Multi-agent) |
| 2 | 4, 5, 6 | 6 | Workflow & skills (SDD → Vibe → Claude Skills) |
| 3 | 7, 8, 9 | 6 | Alt arch + Files + Custom Graph |
| 4 | 10, 11, 12, 13 | 8 | Production (Eval → Cost → Streaming → Deploy) |
| 5 | 14, 15, 16 | 6 | Architect skills (Interview → Red-team → Product) |
| 6 | 17, 18, 19 | 6 | **Healthcare deep dive** |
| 7 | 20, 21, 22 | 6 | **Agriculture deep dive** |
| 8 | 23, 24, 25 | 6 | **Finance deep dive** |
| 9 | 26, 27, 28 | 6 | **Vidya Karana** (Wellness + Yoga + Vedic) |
| 10 | 29, 30 | 4 | **Family AI** — landscape + meal/archivist |
| 11 | 31, 32 | 4 | **Family AI** — scheduler/proactive + capstone build |

Total: ~64 hours over 11 weeks. The 9-hour weekly budget allows ~1 hour/week of buffer for review, reading, and life.

---

## Per-session detail (open the CSV for full data)

The CSV has 11 columns:

| Column | Purpose |
|---|---|
| **Session** | Session number (1-32) |
| **Week** | Calendar week (1-11) |
| **Suggested Days** | Recommended days within the week — adjust to your schedule |
| **Track** | Track letter + name (A through L) |
| **Title** | Session title |
| **Hours** | Estimated duration |
| **Status** | **You fill in:** "Not Started" → "In Progress" → "Done" |
| **Files to Build** | The artifacts each session produces |
| **Key Patterns / Concepts** | What you'll learn / what to file away |
| **Prerequisites** | Which prior session(s) must be done first |
| **Notes** | Your notes column — for observations, gotchas, follow-ups |

---

## Open clarification items (will refine specific sessions when answered)

1. **Vidya Karana repo** — paste README or make `SreeGD/vidya-karana` public; I'll align Sessions 26-28 to the actual project
2. **Healthcare focus** — US/HIPAA or India/ABDM? (default: India-first with US callouts)
3. **Agriculture focus** — smallholder/India or commercial/global? (default: India smallholder)
4. **Finance sub-vertical** — retail banking / capital markets / payments / insurance? (default: retail banking + capital markets)
5. **Family AI killer scenario** — confirmed: all three (A meal + B archivist + C scheduler/proactive)
6. **Session 1 Python version** — official `mcp` SDK requires Python 3.10+; your current venv is 3.9. Options below.

---

## How to use this curriculum

1. **Open `CURRICULUM.csv` in Excel or Google Sheets** — it's the working tracker
2. **Sort by Session # ascending** to follow the linear path
3. **Filter by Track** to focus on one theme at a time
4. **Update Status column** as you progress
5. **Add your own notes** in the Notes column — these become your architect's notebook
6. **At the end of each session, capture:** what surprised you, what you'd do differently, what to revisit

---

## Open question — Session 1 Python version

The official Anthropic `mcp` SDK requires **Python 3.10+**, your `.venv` is on **3.9.13**. Three options to unblock Session 1:

| Option | Effort | Trade-off |
|---|---|---|
| **Upgrade Python** — install 3.11 / 3.12 (via pyenv, brew, or asdf), recreate the venv, reinstall all requirements | ~30 min | Cleanest path; needed eventually anyway |
| **Build MCP demo without the SDK** — implement JSON-RPC over stdio by hand (~80 lines of Python) | ~30 min | More pedagogically educational, less practical |
| **Skip Session 1** — start with Session 2 (Reflection + Plan-and-Execute) which has no version constraint | 0 min | Defers MCP to later; rest of curriculum proceeds |

My recommendation: **upgrade Python** — it's a one-time investment that also unlocks newer language features used by other libraries down the road.
