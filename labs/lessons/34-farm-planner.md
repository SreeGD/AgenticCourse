# 34 — Suryapet Farm Planner (Session 22)

> **The largest single session in the curriculum so far — a small product, not a lab demo.** Knowledge-grounded farm-planning advisor for Telangana's Suryapet / Jangaon / Nalgonda triangle: pick a farmer profile, set goals + constraints, and the LLM generates a multi-year crop plan with mixed-farming options (dairy, apiary, poultry, fish, sericulture, mushroom), exotic crop integration (avocado, dragon fruit, pomegranate, etc.), sustainability practices, a 10-year cash flow, and concrete next steps — wrapped in a Streamlit UI with PDF export and a FastAPI stub for the future React frontend.

---

## Roadmap — where this lesson sits in the journey

```
═══════ PHASE 1: FOUNDATION (done) ═══════                ═══════ PHASE 3: VERTICALS ═══════

  ✓ 01-21 (foundation + RAG + production + architect)      Track H: AGRICULTURE
                                                              ▶ Session 22: FARM PLANNER  ◄ HERE
                                                              ○ Session 23: Crop Diagnostic (vision)
                                                              ○ Session 24: Vernacular Bot (WhatsApp)
                                                            Track J: ○ Finance
                                                            Track K: ○ Vidya Karana
                                                            Track L: ○ Family AI
                                                            Track M: ○ Claude Code Mastery
```

**Why this lesson now:** Phase 1 + 2 built the LLM toolkit (LangChain, RAG, eval, cost, governance, UX). Phase 3 turns the toolkit on real vertical domains. Agriculture is the chosen first vertical because the three constraints — multi-modal, offline-tolerant, ₹-paise unit economics — force the most interesting architectural choices in the entire curriculum.

This session focuses on the **knowledge-grounded LLM pattern**: embed 7-8K tokens of regional expertise into the system prompt so the advisor stops giving generic "grow tomatoes" answers and starts giving real Suryapet-specific guidance with variety-level economics, govt scheme references, and supplier contacts.

---

## Files involved

| File | Role |
|---|---|
| [`34_farm_planner_engine.py`](../34_farm_planner_engine.py) | **Pure Python engine** — no UI imports. All business logic: profile + plan I/O, LLM call with knowledge-base-grounded system prompt, sustainability scoring (deterministic), markdown + PDF rendering. Reusable from any UI. |
| [`34_farm_planner_ui.py`](../34_farm_planner_ui.py) | **Streamlit UI** — multi-page form-driven app. Calls engine functions; no business logic of its own. Run with `streamlit run 34_farm_planner_ui.py`. |
| [`34_farm_planner_api.py`](../34_farm_planner_api.py) | **FastAPI stub** — REST endpoints wrapping the same engine. Proves the future React-frontend migration path is a swap, not a rewrite. Run with `uvicorn 34_farm_planner_api:app`. |
| [`agritech/landscape.md`](../agritech/landscape.md) | Slim AgriTech AI landscape (stakeholders, why this is uniquely demanding, use-case taxonomy) |
| [`agritech/telangana_knowledge_base.md`](../agritech/telangana_knowledge_base.md) | **The 7K-token knowledge base** embedded into the system prompt. Districts, soils, variety economics, mixed farming, sustainability, govt schemes, suppliers, market channels, wildlife matrix, sustainability scoring rubric. |
| [`farm_profiles/sample_*.json`](../farm_profiles/) | Three sample profiles — Suryapet (5 ac, mixed regenerative), Jangaon (8 ac, black cotton commercial), Nalgonda (12 ac, alluvial canal command commercial+perennials). |

---

## What problem it solves

Default LLM farm advisors fail in three predictable ways:

1. **Climate-blind recommendations.** "Try Hass avocado!" — except Suryapet summer hits 38-40°C, which kills Hass. Fuerte / Pollock / Ettinger are the green-skin varieties that work; a generic LLM doesn't know to recommend them.

2. **Variety-agnostic economics.** "Lemon is profitable!" — except Thailand Lemon fetches +40-50% vs Kagzi, Vikram Seedless +60-80%, Pramalini is the disease-resistant backup. Variety-level guidance is where the actual value is, and generic LLMs operate at crop level.

3. **No regional supply chain knowledge.** "Sell at the local mandi!" — except Jangaon's primary wholesale is Warangal (60 km), Nalgonda's is Miryalaguda, Suryapet routes to Hyderabad Monda (120 km). And the suppliers (SKLTSHU Rajendranagar, Deccan Exotics in Kukatpally, Indo Israel for Ashdot 17 rootstock) are public-domain regional knowledge that's missing from default training.

The advisor fixes all three by embedding the [Telangana Knowledge Base](../agritech/telangana_knowledge_base.md) into the system prompt (with `cache_control: ephemeral` so repeated planning runs are cheap). Same LLM, dramatically better output.

---

## The analogy

**A doctor with the local case notebook vs a doctor with only textbook knowledge.**

A doctor at a tertiary-care city hospital has textbook knowledge of cardiac disease. A district doctor in Nalgonda has the same textbook knowledge *plus* the local case notebook: "this kind of MI pattern is unusual in our patient demographic," "the nearest cath lab is 80 km in Hyderabad, plan retrieval accordingly," "this medication has supply issues in the district pharmacies — use the alternative."

Same training. Different outcomes. The local notebook is the difference.

The Telangana Knowledge Base is the local case notebook for farm planning. The LLM has the textbook (training data); the knowledge base is the Suryapet-specific notebook.

---

## Visual

```
┌────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT UI                                │
│                                                                        │
│  Home → Farm Profile → Goals & Constraints → Generate Plan             │
│             ↓                ↓                       ↓                 │
│   farm_profiles/      session state           View Plan + Download     │
│   sample_*.json                               (Markdown / PDF)         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ calls engine functions
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  ENGINE (34_farm_planner_engine.py)                    │
│                                                                        │
│  generate_farm_plan(profile, goals) ─┐                                 │
│                                      ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  System prompt (with cache_control: ephemeral)                  │  │
│  │   • 7K-token Telangana Knowledge Base inlined                   │  │
│  │   • Per-district: Suryapet / Jangaon / Nalgonda                 │  │
│  │   • Variety economics + suppliers + govt schemes                │  │
│  │   • Sustainability rubric + wildlife matrix                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              +                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  User prompt = profile JSON + goals JSON + planning directive   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                         │
│  claude-sonnet-4-6 with_structured_output(FarmPlan)                    │
│  max_tokens=8192, timeout=240s, retry x3 with exponential backoff      │
│                              ↓                                         │
│  FarmPlan (Pydantic):                                                  │
│    - crops[] (variety-level, with confidence)                          │
│    - livestock[] (dairy / poultry / fish)                              │
│    - apiary (species + boxes + placement strategy)                     │
│    - sustainability_practices[]                                        │
│    - year_by_year_cash_flow[] (10 years)                               │
│    - subsidies + suppliers + market channels                           │
│    - immediate_next_steps + pilot recommendation                       │
│                              ↓                                         │
│  score_sustainability(plan) → composite + 5-axis breakdown             │
│  (deterministic, no LLM)                                               │
│                              ↓                                         │
│  save_plan() → farm_plans/<farmer_id>/<plan_id>.json                   │
│  render_plan_markdown() / render_plan_pdf() — both available           │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ same engine functions
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│           FASTAPI STUB (34_farm_planner_api.py — future)               │
│                                                                        │
│  POST /profile · GET /profile/{id} · POST /plan · GET /plan/{id}.pdf   │
│  → wraps same engine functions as REST. Swap UI to React when ready.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Concept walk-through

### 1. The knowledge base is the system prompt

```python
SYSTEM_PROMPT_TEMPLATE = """You are a senior farm-planning advisor for the
Suryapet / Jangaon / Nalgonda region of Telangana, India...

KNOWLEDGE BASE (authoritative, do not contradict):
==================================================
{knowledge_base}
==================================================
"""

def _build_system_prompt() -> list[dict]:
    knowledge = _load_knowledge_base()
    return [{
        "type": "text",
        "text": SYSTEM_PROMPT_TEMPLATE.format(knowledge_base=knowledge),
        "cache_control": {"type": "ephemeral"},   # Session 4 + 15 caching
    }]
```

The 7K-token knowledge base is inlined into the system prompt with `cache_control: ephemeral`. First call writes the cache (25% premium); every subsequent call within 5 minutes reads the cache at ~10% of the fresh-input price. At scale this is the difference between a sustainable per-query cost and one that doesn't work.

### 2. Pydantic schemas are the contract

```python
class CropInPlan(BaseModel):
    crop_name: str
    variety: str | None
    local_name: str | None
    role: Literal["short_term_cash_crop", "medium_term_crop",
                  "perennial_anchor", "intercrop", "boundary_crop"]
    acres_allocated: float
    time_to_first_yield_years: float
    peak_production_year_start: int
    peak_production_year_end: int
    revenue_per_acre_at_peak_inr: str
    year_1_investment_inr: str
    breakeven_year: int
    # ... 20+ fields total
    is_exotic_high_value: bool
    pollinator_friendly: bool
    confidence_self: float
    confidence_meta: float
```

The schema *forces the LLM to generate variety-level detail*. Without a `variety` field, the LLM would say "lemon"; with the field, it says "Thailand Lemon" because the schema asks for it. The `confidence_self` and `confidence_meta` fields (Session 20 pattern) carry honest uncertainty through to the UI.

### 3. The engine is a swap point, not a tangle

```python
# Public API of 34_farm_planner_engine.py
list_profiles() -> list[ProfileSummary]
load_profile(farmer_id) -> FarmProfile
save_profile(profile) -> Path
delete_profile(farmer_id) -> None

generate_farm_plan(profile, goals) -> FarmPlan
score_sustainability(plan) -> SustainabilityScore
save_plan(plan) -> Path
load_plans_for_farmer(farmer_id) -> list[PlanSummary]

render_plan_markdown(plan) -> str
render_plan_pdf(plan, path) -> Path
```

Streamlit UI imports these. FastAPI stub imports these. A future React frontend will call the FastAPI endpoints which call these. Tests would import these directly. **The engine never imports anything from the UI side.** That's the architecture decision that makes the Streamlit→React swap painless.

### 4. Sustainability scoring is deterministic post-LLM

```python
def score_sustainability(plan: FarmPlan) -> SustainabilityScore:
    practices = {p.practice for p in plan.sustainability_practices}

    # 1. Soil health (0-20)
    soil = 0.0
    if "crop_rotation" in practices: soil += 5
    if any(p in practices for p in ("intercropping", "cover_crops")): soil += 5
    if any(p in practices for p in ("composting", "vermicomposting")): soil += 5
    if "zbnf_practices" in practices: soil += 5
    soil = min(soil, 20)
    # ... 4 more axes
```

The score is computed from the plan structure deterministically — not by asking the LLM. This matters because:
- **Reproducible**: same plan → same score, every time
- **Cheaper**: no extra LLM call
- **Auditable**: the rubric is in the code, not in a prompt
- **Tuneable**: change the weights and re-score every saved plan instantly

The composite is 0-100; per-axis scores out of 20; recommendations to lift the score are surfaced as concrete actions.

### 5. The Streamlit → FastAPI migration path

`34_farm_planner_api.py` is ~150 lines of FastAPI that wraps the same engine functions as REST endpoints:

```python
@app.post("/plan", response_model=FarmPlan)
def generate_plan(req: GeneratePlanRequest) -> FarmPlan:
    plan = engine.generate_farm_plan(req.profile, req.goals)
    if req.save:
        engine.save_plan(plan)
    return plan

@app.get("/plan/{farmer_id}/{plan_id}.pdf")
def get_plan_pdf(farmer_id: str, plan_id: str):
    plan = engine.load_plan(farmer_id, plan_id)
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        out = Path(tmp.name)
    engine.render_plan_pdf(plan, out)
    return FileResponse(out, media_type="application/pdf", ...)
```

When you decide to swap UI from Streamlit to React/mobile:
1. Run this file in production via uvicorn (Session 17 deploy pattern)
2. Expand with auth + Postgres (replace JSON file storage)
3. Frontend talks to the API; engine unchanged

Streamlit keeps running locally for power users / KVK staff. Both UIs can coexist on the same engine.

---

## Run it

### Streamlit UI (recommended for first run)

```bash
cd labs
./.venv/bin/python -m streamlit run 34_farm_planner_ui.py
```

Browser opens at `http://localhost:8501`. Pick `sample_suryapet` from the Home page → look at the auto-loaded profile → go to Goals & Constraints → tick `Include dairy` and `Include apiary` → Generate Plan → watch the spinner (~30-120s) → View Plan → explore tabs → Download as PDF.

### FastAPI stub (future migration preview)

```bash
cd labs
./.venv/bin/python -m uvicorn 34_farm_planner_api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI. Try:
- `GET /profile` to list samples
- `GET /profile/sample_suryapet` to see the JSON
- `GET /health` for liveness/readiness
- `POST /plan` with a body containing profile + goals to generate

### Engine sanity check

```bash
./.venv/bin/python 34_farm_planner_engine.py
```

Prints: knowledge base size, profile directory, list of loaded profiles. No LLM call.

---

## What a generated plan looks like (real output)

A live run against the Suryapet sample profile (5 acres, red soil + chalka, partial irrigation via 1 bore well, monkey + peacock pressure, transitioning to organic, ₹4 lakh investment capacity) with goals = `diversification_resilience`, dairy + apiary included, exotic crops (avocado, dragon fruit, moringa) of interest, produces a plan that typically looks like:

**Plan Summary** (LLM-generated, one paragraph)
> A 5-acre integrated regenerative mix: 1.5 acres Thailand Lemon as the perennial anchor with Year-4 first yield and Year-8+ peak income; 1 acre pulses + millets rotation for short-term cash flow Years 1-2 while perennials develop; 0.5 acre Fuerte avocado pilot (per Telangana Hort Dept guidance to start small); 1 acre vegetable + groundnut rotation; drumstick + neem boundary plantings for wildlife defense and pollinator support; 1 indigenous Sahiwal dairy cow integrated for manure + monthly income; 4 Apis mellifera hives placed within 100m of lemon + drumstick blooms. ZBNF transition over Years 1-3.

**Crops (5-6 typically)**
- Lemon — *Thailand Lemon* (1.5 ac, perennial_anchor, Y1 invest ₹65K, Y8+ peak ₹50K-1L/ac, breakeven Y4)
- Pigeonpea — *Redgram* (0.75 ac, short_term_cash_crop, ₹20K Y1, ₹25-40K/ac/cycle)
- Ragi — *finger millet* (0.5 ac, short_term_cash_crop, ₹12K Y1, ₹15-25K/ac, monkey-resistant)
- Avocado — *Fuerte* (0.5 ac, perennial_anchor PILOT, ₹1.5L Y1, ₹10-13L/ac at Y5+)
- Drumstick — *PKM-1* (boundary, ₹8K total, ₹50K-1L/ac/yr at maturity, wildlife-proof)
- Groundnut/vegetables rotation (1 ac, ₹40-60K Y1, monthly cash)

**Livestock**
- Sahiwal cow ×1 — 8-12 L/day milk, ₹5-12K/mo net, manure for ~1 acre, NLM 25-50% subsidy on indigenous breed

**Apiary**
- Apis mellifera × 4 boxes — 25 kg/box/yr honey (₹25K/yr revenue), 25-40% pollination boost to lemon + drumstick + groundnut, MIDH ₹2K/box subsidy

**Sustainability Practices (~5)**
- ZBNF transition (₹0 incremental, 70-80% input cost cut by Y3)
- Vermicomposting (₹15K setup, 6-month payback via reduced fertilizer)
- Drip irrigation (already in place; 80% MIDH subsidy if expanding)
- Mulching (free from on-farm residues)
- Agroforestry (drumstick + neem boundaries; carbon credit potential)

**10-Year Cash Flow** (table form)
- Y1: invest ₹4.0 L · revenue ₹50K (early pulses + dairy) · net **−₹3.5 L**
- Y2: invest ₹40K · revenue ₹1.5L (pulses + millet + dairy) · net ₹1.1L
- Y4: lemon first yield · net ₹1.8L
- Y8+: peak · net **₹4-5 L/yr** (lemon mature + avocado bearing + dairy + apiary)

**Sustainability score**: typically 65-80/100 with this profile mix — score climbs as ZBNF establishes (Year 2+) and biogas/solar pump get added.

**Govt subsidies to pursue**: MIDH drip + planting material · MIDH apiary subsidy · NLM indigenous breed · Rythu Bandhu ₹50K/yr (5 ac × ₹10K) · PMFBY crop insurance

**Suppliers to contact**: SKLTSHU Rajendranagar (lemon + avocado saplings) · Deccan Exotics Hyderabad · Indo Israel Avocado (Ashdot 17 rootstock) · KVK Suryapet (apiary training)

**Pilot recommendation**: Start avocado at 0.5 ac before scaling; observe Year 1-2 establishment success; expand to 1+ ac in Year 3 if growth and disease pressure stay manageable.

**Disclaimers**: Validate with Suryapet KVK officer before planting. ₹ projections depend on market, weather, management quality. Wildlife pressure varies year-on-year; reassess deterrents annually.

> *Note: live output varies per run as the LLM rebalances the plan. The shape (variety-level granularity, multi-year cash flow, govt-scheme references, supplier names, sustainability scoring) is consistent because the schema enforces it.*

---

## Production patterns

### When to embed knowledge vs RAG

This session embeds the entire 7K-token knowledge base in the system prompt. That works because:
- The base is small enough to fit
- It changes slowly (district-level facts, not daily prices)
- It's used in every call (so cache_control pays off)

**Switch to RAG when** the knowledge base grows past ~30K tokens, or when different queries need different subsets (e.g., crop-specific datasheets, weather data, mandi price feeds). RAG retrieves only the relevant chunks per query; embedded is whole-base-every-call.

The pattern transfers: Session 9 (RAG) + Session 11 (Hybrid RAG) + Session 12 (GraphRAG) are the alternatives when this approach doesn't scale.

### Engine separation pays off repeatedly

The strict UI-engine separation is a one-time architectural decision that buys you:
- Testability (pytest the engine directly, no Streamlit fixture)
- Multi-UI support (Streamlit + FastAPI today, React tomorrow, mobile next month)
- Refactor safety (changing rendering doesn't touch business logic)
- Type safety (Pydantic schemas at the boundary catch errors at validation, not deep in the call stack)

The cost: zero runtime overhead. Just discipline.

### Sustainability scoring is deterministic by design

The composite score is **NOT** computed by the LLM. It's a deterministic function of the plan structure. This means:
- Auditors can replay the score on saved plans
- Score changes only when the rubric or the plan changes
- No LLM-judge non-determinism contaminates the metric

Same pattern as Session 14 (eval) — LLM generates content, deterministic Python computes metrics.

### The retry + backoff loop is mandatory

Anthropic's API has occasional overload (529), connection drops, and timeouts at 30-60s intervals. The engine's retry loop with exponential backoff (0s, 10s, 30s, 60s between retries) handles transients without manual intervention. Production should also:
- Surface "retrying due to overload..." to the user (Session 21 UX pattern)
- Track retry rates as a metric (Session 17 observability)
- Fail open to a manual-review queue if retries exhaust

### PDF generation choice — fpdf2

Picked `fpdf2` because:
- Pure Python (no system deps like cairo/pango)
- Already in `requirements.txt` from earlier sessions
- Adequate for tabular plan output (cash flow tables, crop summaries)

`weasyprint` would render markdown→HTML→PDF with prettier typography but needs cairo + pango system libraries. Tradeoff: ship simpler today, swap later if needed.

### Knowledge base hygiene

Don't commit private farmer data. The `farm_profiles/.gitignore` excludes everything except `sample_*.json`. The sample profiles use fictional farmer names + sanitized data. Real Suryapet farmer profiles stay on the farmer's (or KVK officer's) local machine.

---

## Try this

1. **Edit `sample_suryapet.json`** to add 2 acres of black cotton soil. Re-run the planner. Does it now recommend raised beds for perennials? (It should — the knowledge base says perennials on black cotton need raised beds to prevent Phytophthora.)

2. **Toggle `include_sericulture: True`**. Run a plan. The advisor should now consider mulberry sericulture as a possible income stream — but only if labor is available (sericulture is labor-intensive). Confirm the plan respects the labor constraint.

3. **Set `organic_required: True`** on the Nalgonda profile. The plan should drop any crop that's hard to grow organic (cotton, intensive vegetables) and emphasize ZBNF transition, organic-friendly crops, and the PGS-India certification path.

4. **Add a new district** (e.g., Mahbubnagar) to the knowledge base. Update the district table with rainfall + soil + market + KVK. The advisor should now produce plans for that district without code changes. This proves the knowledge-grounded pattern scales.

5. **Build a `pytest` against the engine.** Test: profile round-trip (save → load → equal), sustainability scoring on a hand-built plan (known practices → known score), PDF rendering doesn't crash on a sample plan. Engines that aren't UI-coupled are unit-testable.

6. **Wire a real mandi price API** (Agmarknet or CommodityOnline). Add a tool that fetches today's price for the recommended crop variety. The advisor can then layer "current market = ₹X/kg above/below the long-term average" on top of the static economics.

7. **Run the FastAPI stub.** `uvicorn 34_farm_planner_api:app --reload`. Hit it from `curl`:
   ```bash
   curl -s http://localhost:8000/profile | jq .
   curl -X POST http://localhost:8000/plan -H "Content-Type: application/json" \
        -d '{"profile": {...}, "goals": {...}}' | jq .crops
   ```
   Compare the output to the Streamlit UI's rendering. Same engine, two transports.

8. **Add a new mixed-farming option** — say, mushroom cultivation. Extend the `PlanningGoals.include_mushroom` flow, add a `MushroomInPlan` schema to the engine, update the knowledge base section on mushroom (oyster ₹20-30K setup, 30-day cycle, urban demand). The advisor will start integrating mushroom in plans where it fits.

---

## Mental model

> **Vertical AI = generic LLM + knowledge base + structured output. The knowledge base is the moat; the LLM is interchangeable.**

Three slogans:

1. **"The local case notebook beats the textbook."** Embed regional expertise into the system prompt; outputs become district-specific instead of generic.
2. **"Engine separation pays for itself the first time you want a new UI."** Streamlit today, React tomorrow, mobile next month — the engine doesn't care.
3. **"Deterministic scoring + LLM generation = auditable AI."** LLMs generate the content; deterministic Python computes metrics on top. Auditors and regulators can replay the scoring forever.

---

## FAQ

**Q: Why Streamlit when the lesson says it's a "future migration to FastAPI"?**
Streamlit gets you a usable UI in ~600 LoC of form code. FastAPI + React would be 5x the code for the same product. The architecture (engine separation) means you can start with Streamlit, validate the product, then migrate the UI without rewriting business logic. Most production AI features should start this way.

**Q: Does the LLM hallucinate variety names or supplier contacts?**
The knowledge base explicitly lists varieties (Thailand Lemon, Fuerte avocado, Bhagwa pomegranate...) and suppliers (SKLTSHU, Deccan Exotics, Indo Israel Avocado). The system prompt instructs the LLM to use these — not invent. In practice you'll see occasional drift; the disclaimer + KVK escalation note is the safety net.

**Q: How does this differ from Session 12 (GraphRAG)?**
GraphRAG retrieves a subgraph at query time based on entities in the query. This session embeds the *whole* knowledge base every call. The trade-off: GraphRAG scales to larger knowledge bases; embedding is simpler and cheaper for small (≤30K token) bases. For ~7K tokens of regional farm knowledge, embedding wins.

**Q: Why three sample profiles instead of one?**
Each district has different soil + climate + market structure. A planner that works for Suryapet but fails on Jangaon's black cotton soil isn't generalizable. Three profiles stress-test the advisor across the dominant variations in the region.

**Q: How big can the schema get before the LLM starts hallucinating fields?**
At this schema (CropInPlan has ~20 fields, FarmPlan has nested CropInPlan + LivestockInPlan + ApiaryInPlan + ...), max_tokens=8192 and timeout=240s are needed. Larger schemas push past Sonnet's effective context for structured output. Production move: split into multiple calls (one for crops, one for cash flow, one for sustainability) and stitch together. Same pattern as multi-step agents (Session 3 / Session 13).

**Q: Why timeout=240 and max_tokens=8192?**
The full FarmPlan output is ~3K-5K output tokens (60+ fields total, including nested lists). At Sonnet's generation speed (~50-100 tokens/sec for structured output), that's 30-90 seconds. The 240s timeout gives headroom for network jitter + Anthropic load. max_tokens=8192 prevents the response from being truncated mid-generation.

**Q: How do I add a new crop to the knowledge base?**
Edit `agritech/telangana_knowledge_base.md`. Add a section under the right category (perennial / annual / exotic / mixed farming). Include: varieties, time to bearing, ₹/acre economics, soil + climate fit, suppliers, govt schemes, disease risks, market channels. The advisor picks it up on the next planning call (the knowledge base is reloaded each call; not import-time cached). No code changes needed.

**Q: Can the advisor recommend crops outside the knowledge base?**
Yes, the LLM can draw on training knowledge for crops not explicitly in the base. But confidence will be lower (the calibration guide in the KB says experimental crops get <0.65 confidence). The UI surfaces confidence, so users see when the advisor is reaching beyond authoritative knowledge.

**Q: What about offline operation?**
The advisor needs the Anthropic API to generate plans. The UI is local, the JSON state is local, but the LLM call is remote. For truly offline operation, you'd need a local LLM (llama.cpp + Qwen-32B or similar) — quality will drop but it works. Mention as a future extension; Session 24 (vernacular bot) revisits offline patterns.

**Q: How accurate are the ₹ projections?**
Calibrated to public market intelligence circa 2026 (the lemon report + avocado guide from the knowledge base). Ranges are wide intentionally (₹2-4 lakh, not ₹2.347 lakh) because real outcomes vary 30-50% depending on management quality + weather + market. The advisor returns ranges; users (and KVK officers) calibrate based on local context.

**Q: How does this scale to other states / regions?**
Build a knowledge base per region. Replace `agritech/telangana_knowledge_base.md` with `agritech/maharashtra_knowledge_base.md` (or whichever region) and the advisor immediately works there. The engine is region-agnostic; only the system prompt content is region-specific. This is the productization path: one engine, many regional knowledge packs.

**Q: What about Anthropic API overload (529 errors)?**
The retry loop in `generate_farm_plan` handles 529s with exponential backoff (0s, 10s, 30s, 60s). If all 3 retries fail, the error propagates. Production should: (1) surface "we're temporarily overloaded" to the user, (2) queue the request for async processing, (3) increase backoff for the next user during the same overload window. Same as Session 17's circuit breaker pattern.

**Q: How does PDF generation work, and what are its limits?**
`fpdf2` is pure Python; renders text + tables + basic layout. Limits: no Unicode support without bundled TTF fonts (we sanitize ₹ → `Rs.`, → → `->`, etc. via the `_safe()` helper). For prettier output, swap to `weasyprint` (HTML→PDF) or `reportlab` — both heavier, both produce nicer PDFs. The lab's fpdf2 output is functional, not beautiful.

**Q: Is the FastAPI stub production-ready?**
No. It's a *demonstration* of the architecture. Production needs: auth (Session 18 patterns), persistence in a real DB (not JSON files), observability (Session 17), rate limiting, audit logging (Session 20), governance (Session 19). All the Track F + G patterns layer on top of the same engine.

**Q: What's the next step in Track H?**
Session 23 — Crop Diagnostic + Advisory (vision-first agent). Take a photo of a diseased crop; the agent identifies the disease and recommends treatment in vernacular language. Multi-modal (image + Telugu text), offline-tolerant (edge inference for triage), and integrates with the farm plan ("the disease you have is X; your current plan has these crops at risk").

---

## Related

- **Previous:** [33 — UX Patterns](33-ux-patterns.md) (Track G complete)
- **Next:** Session 23 — Crop Diagnostic + Advisory (vision-first agent for Telugu-speaking farmers)
- **Builds on:** [04 — Prompt caching](04-prompt-caching.md) (cache_control on the knowledge base), [05 — Structured output](05-structured-output.md) (with_structured_output for the FarmPlan schema), [09 — RAG](09-rag.md) (the knowledge-grounding alternative), [14 — Multi-agent + LTM](14-multi-agent-ltm.md) (saved profiles + plans = memory), [17 — Deploy + Observability](28-production-deploy.md) (FastAPI + uvicorn pattern), [20 — Governance](32-governance.md) (confidence scoring per recommendation), [21 — UX Patterns](33-ux-patterns.md) (disclaimers, escalation to KVK, trust calibration)
- **Track H status:** ▶ 1/3 complete. Next: Crop Diagnostic (vision) → Vernacular Bot (WhatsApp + Telugu).
