"""FastAPI stub — REST API wrapper over the same engine the Streamlit UI uses.

This file exists TODAY to prove the architecture: when you swap the
Streamlit UI for a React frontend later, the engine is unchanged; you
just expand this file with auth + persistence + deployment (Session 17).

Run locally:
    ./.venv/bin/python -m uvicorn 34_farm_planner_api:app --reload --port 8000

Endpoints:
    POST   /profile                   create profile
    GET    /profile/{farmer_id}       retrieve profile
    GET    /profile                   list all profiles
    PUT    /profile/{farmer_id}       update profile
    DELETE /profile/{farmer_id}       delete profile

    POST   /plan                      generate a plan (body: profile + goals)
    GET    /plan/{farmer_id}/{plan_id} retrieve a saved plan
    GET    /plan/{farmer_id}          list plans for a farmer

    POST   /sustainability/score      score a given plan
    GET    /plan/{farmer_id}/{plan_id}.md  download plan as markdown
    GET    /plan/{farmer_id}/{plan_id}.pdf download plan as PDF

    GET    /health                    liveness + readiness
"""

import importlib.util
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

# Import the engine via importlib (file starts with digit so can't be a normal import)
_engine_path = Path(__file__).parent / "34_farm_planner_engine.py"
_spec = importlib.util.spec_from_file_location("engine", _engine_path)
engine = importlib.util.module_from_spec(_spec)
sys.modules["engine"] = engine
_spec.loader.exec_module(engine)

# Re-export the schemas for FastAPI's docs
FarmProfile = engine.FarmProfile
PlanningGoals = engine.PlanningGoals
FarmPlan = engine.FarmPlan
SustainabilityScore = engine.SustainabilityScore
ProfileSummary = engine.ProfileSummary
PlanSummary = engine.PlanSummary

app = FastAPI(
    title="Suryapet Farm Planner API",
    version="0.1.0",
    description="REST API wrapper over the farm-planner engine. "
                "Same engine drives the Streamlit UI in 34_farm_planner_ui.py.",
)


# =====================================================================
# Profile endpoints
# =====================================================================

@app.post("/profile", response_model=FarmProfile, tags=["profile"])
def create_profile(profile: FarmProfile) -> FarmProfile:
    """Create a new farm profile. Use engine.make_farmer_id() for a new ID."""
    if not profile.farmer_id:
        profile.farmer_id = engine.make_farmer_id()
    engine.save_profile(profile)
    return profile


@app.get("/profile", response_model=list[ProfileSummary], tags=["profile"])
def list_profiles_endpoint() -> list[ProfileSummary]:
    return engine.list_profiles()


@app.get("/profile/{farmer_id}", response_model=FarmProfile, tags=["profile"])
def get_profile(farmer_id: str) -> FarmProfile:
    try:
        return engine.load_profile(farmer_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Profile {farmer_id} not found") from e


@app.put("/profile/{farmer_id}", response_model=FarmProfile, tags=["profile"])
def update_profile(farmer_id: str, profile: FarmProfile) -> FarmProfile:
    if profile.farmer_id != farmer_id:
        raise HTTPException(status_code=400, detail="Path / body farmer_id mismatch")
    engine.save_profile(profile)
    return profile


@app.delete("/profile/{farmer_id}", tags=["profile"])
def delete_profile_endpoint(farmer_id: str) -> dict:
    engine.delete_profile(farmer_id)
    return {"status": "deleted", "farmer_id": farmer_id}


# =====================================================================
# Plan endpoints
# =====================================================================

class GeneratePlanRequest(BaseModel):
    profile: FarmProfile
    goals: PlanningGoals
    save: bool = True


@app.post("/plan", response_model=FarmPlan, tags=["plan"])
def generate_plan(req: GeneratePlanRequest) -> FarmPlan:
    """Generate a farm plan. The LLM call lives here.
    Latency ~30-60s. Production should make this async or stream.
    """
    plan = engine.generate_farm_plan(req.profile, req.goals)
    if req.save:
        engine.save_plan(plan)
    return plan


@app.get("/plan/{farmer_id}", response_model=list[PlanSummary], tags=["plan"])
def list_plans(farmer_id: str) -> list[PlanSummary]:
    return engine.load_plans_for_farmer(farmer_id)


@app.get("/plan/{farmer_id}/{plan_id}", response_model=FarmPlan, tags=["plan"])
def get_plan(farmer_id: str, plan_id: str) -> FarmPlan:
    try:
        return engine.load_plan(farmer_id, plan_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found") from e


@app.get("/plan/{farmer_id}/{plan_id}.md", response_class=PlainTextResponse, tags=["plan"])
def get_plan_markdown(farmer_id: str, plan_id: str) -> str:
    try:
        plan = engine.load_plan(farmer_id, plan_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found") from e
    return engine.render_plan_markdown(plan)


@app.get("/plan/{farmer_id}/{plan_id}.pdf", tags=["plan"])
def get_plan_pdf(farmer_id: str, plan_id: str):
    try:
        plan = engine.load_plan(farmer_id, plan_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found") from e
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        out = Path(tmp.name)
    engine.render_plan_pdf(plan, out)
    return FileResponse(str(out), media_type="application/pdf",
                        filename=f"farm_plan_{plan_id}.pdf")


# =====================================================================
# Sustainability + health
# =====================================================================

@app.post("/sustainability/score", response_model=SustainabilityScore, tags=["sustainability"])
def score_plan(plan: FarmPlan) -> SustainabilityScore:
    return engine.score_sustainability(plan)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness (process up) + readiness (knowledge base loaded)."""
    kb_exists = engine.KNOWLEDGE_BASE_PATH.exists()
    profiles_count = len(engine.list_profiles())
    return {
        "status": "ok" if kb_exists else "degraded",
        "liveness": True,
        "readiness": kb_exists,
        "knowledge_base_path": str(engine.KNOWLEDGE_BASE_PATH),
        "profiles_on_disk": profiles_count,
    }


# =====================================================================
# Migration notes (when you swap from Streamlit to React frontend)
# =====================================================================
#
# 1. Add auth — JWT middleware (FastAPI security utilities)
# 2. Replace filesystem JSON with Postgres:
#       engine.save_profile / load_profile / list_profiles
#    are the swap points. Replace their bodies; signatures unchanged.
# 3. Make /plan async + streaming:
#       async def generate_plan_stream(req) -> StreamingResponse:
#           async for chunk in engine.generate_farm_plan_stream(...):
#               yield chunk
#    The engine would gain an async + streaming entry point.
# 4. Deploy via Session 17 patterns (Dockerfile + fly.toml + observability).
# 5. Add rate limiting, audit logging (Session 19, 20 patterns).
