"""EcoNav AI — OpenEnv-Compliant FastAPI Server.

Mounts OpenEnv RL endpoints at root level (/reset, /step, /state, /grade, /tasks)
and serves the React frontend from /app.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath("."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from packages.env_core.environment import TASKS, ExposureCreditEnv
from packages.env_core.models import ResetRequest, StepRequest
from packages.env_core.aqi_engine import (
    CITY_PROFILES,
    ADJACENCY,
    EDGE_DISTANCES,
    get_city_name,
    _get_aqi_for_city,
    get_grade_for_aqi,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("EcoNav AI Environment starting...")
    print(f"  -> {len(TASKS)} tasks loaded")
    print(f"  -> {len(CITY_PROFILES)} cities in graph")
    print("  -> OpenEnv endpoints ready at /reset /step /state /grade /tasks")
    yield
    print("EcoNav AI shutting down.")


app = FastAPI(
    title="EcoNav AI — Exposure Credit Environment",
    version="1.0.0",
    description="OpenEnv RL environment: navigate Indian cities minimising pollution exposure",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
_env = ExposureCreditEnv()

# ---------------------------------------------------------------------------
# OpenEnv standard endpoints (ROOT level — required by spec)
# ---------------------------------------------------------------------------


@app.post("/reset")
def reset(request: ResetRequest | None = None):
    """Reset the environment and start a new episode."""
    task_id = request.task_id if request else "easy_route"
    try:
        obs = _env.reset(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    """Execute an action (move to a city) and return the new observation."""
    try:
        result = _env.step(request.action)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.get("/state")
def state():
    """Get current episode state/metadata."""
    try:
        s = _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return s.model_dump()


@app.get("/tasks")
def get_tasks():
    """List all available tasks with difficulty levels."""
    return {
        "tasks": [t.model_dump() for t in _env.get_tasks()],
        "count": len(TASKS),
    }


@app.get("/grade")
def grade():
    """Grade the current (completed) episode."""
    try:
        result = _env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get details of a specific task."""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Available: {list(TASKS.keys())}",
        )
    return task.model_dump()


# ---------------------------------------------------------------------------
# Additional API endpoints for the frontend
# ---------------------------------------------------------------------------


@app.get("/api/v1/graph")
def get_graph():
    """Get the full city graph for visualization."""
    cities = []
    for code, profile in CITY_PROFILES.items():
        if code not in ADJACENCY:
            continue
        aqi = _get_aqi_for_city(code)
        g = get_grade_for_aqi(aqi)
        cities.append({
            "code": code,
            "name": profile["name"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "aqi": aqi,
            "grade": g["grade"],
            "label": g["label"],
        })

    edges = []
    for (c1, c2), dist in EDGE_DISTANCES.items():
        edges.append({
            "from": c1,
            "to": c2,
            "from_name": get_city_name(c1),
            "to_name": get_city_name(c2),
            "distance": dist,
        })

    return {"cities": cities, "edges": edges}


@app.get("/")
def root():
    """Serve frontend or show API info."""
    dist_path = os.path.join(os.path.abspath("."), "frontend", "index.html")
    if os.path.exists(dist_path):
        return FileResponse(dist_path)
    return {
        "project": "EcoNav AI — Exposure Credit Environment",
        "version": "1.0.0",
        "openenv": True,
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "grade": "GET /grade",
            "tasks": "GET /tasks",
            "graph": "GET /api/v1/graph",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "EcoNav AI"}


# ---------------------------------------------------------------------------
# Serve frontend static files (if present)
# ---------------------------------------------------------------------------
frontend_dir = os.path.join(os.path.abspath("."), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
