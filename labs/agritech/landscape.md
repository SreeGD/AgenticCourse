# AgriTech AI Landscape — Telangana focus

> The compact reference for "where does AI fit in agriculture, and where doesn't it." Sits alongside the [Telangana Knowledge Base](telangana_knowledge_base.md) which has all the variety + economics detail. This file is about the *stakeholder map and the constraint shape* — the why behind the architecture choices in Sessions 22-24 of the curriculum.

---

## Stakeholders

| Actor | What they need from AI |
|---|---|
| **Smallholder farmer** (1-2 acres, ~80M households in India) | Crop selection, disease diagnosis, pricing, scheme navigation — in their language, on WhatsApp, sub-paise per query |
| **Commercial farmer** (10-50 acres) | Decision support, market timing, multi-crop planning, supplier discovery |
| **FPO (Farmer Producer Org)** | Aggregation, bulk pricing, scheme application help, quality benchmarking |
| **Agri-input retailer** | Variety selection advisory to customers, dosage calculators |
| **Govt extension (KVK)** | Triage tool, decision-support for the ~50 farmers each KVK officer serves |
| **Agri-fintech** | Loan eligibility, crop risk scoring, KCC processing |
| **Crop insurance (PMFBY)** | Claim verification, satellite + ground truth |
| **Agri-trader / mandi** | Quality grading, price discovery |
| **Consumer-facing brands** | Farm-to-consumer traceability, organic certification |

The Suryapet/Jangaon/Nalgonda farm planner (this curriculum's Track H lab) targets the **smallholder + commercial farmer** primarily, with **KVK officer as a power user** secondary.

---

## Why agriculture AI is uniquely demanding

Three constraints rule out 90% of patterns that work in other verticals:

### 1. Multi-modal is a requirement, not a feature

Farmers communicate naturally via:
- **Photos** of diseased plants
- **Voice messages** in vernacular (Telugu, Hindi, Kannada, Marathi)
- **Short text** in code-mixed English-Indic

Text-only English-first design fails on contact with reality. The curriculum addresses this in **Session 23 (Crop Diagnostic + Advisory — vision-first)** and **Session 24 (Vernacular Farmer Bot)**.

### 2. Offline-tolerant

Tier-2 / tier-3 India has intermittent connectivity. Architecture must degrade gracefully:
- Local-first for read-heavy operations (crop info lookups)
- Queue + sync for write-heavy operations (logging field activity)
- Cached responses for common queries

### 3. Unit economics in ₹-paise

A smallholder can't pay $20/mo. The system must work at **sub-paise per query**:
- Aggressive prompt caching (Session 4 + 15 patterns)
- Routing: cheap Haiku for classification, Sonnet only for complex reasoning
- Batched eval via Anthropic Batches API (Session 15) for offline workloads
- Fine-tuned small models (DistilBERT-class) for high-volume routine tasks

The cost ceiling on a smallholder-facing chatbot is roughly **₹0.10 per session** sustainably. That's 10x tighter than the chatbot economics in Session 15.

---

## Use case taxonomy by lifecycle stage

| Stage | Use cases | Stakeholders |
|---|---|---|
| **Pre-season** | Crop selection (Session 22 lab) · soil test interpretation · input planning · scheme application | Smallholder, commercial, KVK |
| **In-season** | Disease/pest diagnosis (Session 23) · advisory · weather alerts · water management | Smallholder, commercial |
| **Post-harvest** | Pricing · mandi info · logistics · storage advice · quality grading | All |
| **Adjacent** | Credit/insurance application · govt scheme navigation · grievance redressal | Smallholder, FPO |

This curriculum builds the **crop selection** use case (Session 22 lab) deeply, then references the others as natural extensions.

---

## Where this curriculum lands

Track H (Agriculture, this track):

| Session | Focus | Build |
|---|---|---|
| **22** | Suryapet farm planner — crop selection with mixed farming + sustainability | Streamlit UI + LLM advisor + Telangana knowledge base + 3 district samples + FastAPI stub for future migration |
| 23 | Crop disease diagnostic — vision-first agent | Image → diagnosis pipeline with confidence display + escalation |
| 24 | Vernacular farmer bot — WhatsApp + Telugu/Hindi | ASR (Whisper) + IndicTrans2 + WhatsApp Business API + cost engineering |

Session 22 (this lab) establishes the **knowledge-grounded LLM pattern** — embedding ~10K tokens of regional expertise into the system prompt so the advisor stops giving generic "grow tomatoes" answers and starts giving real Telangana-specific guidance.

---

## What's in the knowledge base

See [`telangana_knowledge_base.md`](telangana_knowledge_base.md) for the full content. Summary:

- **3 districts** characterized (Suryapet, Jangaon, Nalgonda)
- **Soil typology** (red, chalka, black cotton, alluvial, saline)
- **15+ crops** with variety-level economics (lemon, avocado, dragon fruit, custard apple, pomegranate, mosambi, mango, cotton, paddy, millets, pulses, oilseeds, vegetables)
- **9 exotic / high-value crops** (aloe vera, stevia, lemongrass, passion fruit, fig, dates, moringa, olive)
- **5 mixed farming options** (dairy, apiary, poultry, fish, sericulture, mushroom)
- **8 sustainability practices** (ZBNF, agroforestry, rainwater harvesting, solar, biogas, composting, drip, mulching)
- **12 govt schemes** (PM-KISAN, Rythu Bandhu, PMFBY, MIDH, NLM, PM-KUSUM, …)
- **Suppliers directory** (nurseries, livestock, bee, inputs)
- **Market channels** (mandi, retail, processing, online, export)
- **Wildlife pressure matrix** + mitigations
- **Sustainability scoring rubric** (5 axes × 20 points = 100)
- **Confidence calibration guide** for the advisor's LLM

---

## What this landscape does NOT cover

- Indian agriculture history / policy debate
- Comparative analysis with other geographies (US, EU, China) beyond brief notes
- The full taxonomy of Indian agriculture extension institutions
- Sub-sector deep dives (sugarcane belt, basmati belt, plantation crops) outside Telangana

Those are interesting but beyond a single-session lab. The Telangana focus is the practical scope; expand later as needed.
