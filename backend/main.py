from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AttributionRequest,
    AttributionResponse,
    DEFAULT_DASHBOARD_DIR,
    DefaultConfigResponse,
    FeatureDashboardResponse,
    HealthResponse,
    InterventionRequest,
    InterventionResponse,
)
from backend.service import LoupeBackendService


app = FastAPI(
    title="Loupe Backend",
    description="Backend for SAE token attributions and latent interventions.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = LoupeBackendService()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/defaults", response_model=DefaultConfigResponse)
def defaults() -> DefaultConfigResponse:
    return DefaultConfigResponse()


@app.get("/api/feature-dashboard", response_model=FeatureDashboardResponse)
def feature_dashboard(
    dashboard_dir: str = DEFAULT_DASHBOARD_DIR,
    top_features: int = 20,
    top_concept_scores: int = 40,
    top_tokens: int = 80,
    top_examples: int = 40,
) -> FeatureDashboardResponse:
    try:
        return service.get_feature_dashboard(
            dashboard_dir=dashboard_dir,
            top_features=top_features,
            top_concept_scores=top_concept_scores,
            top_tokens=top_tokens,
            top_examples=top_examples,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/generate-attributions", response_model=AttributionResponse)
def generate_attributions(request: AttributionRequest) -> AttributionResponse:
    try:
        return service.generate_attributions(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {exc}") from exc


@app.post("/api/interventions", response_model=InterventionResponse)
def interventions(request: InterventionRequest) -> InterventionResponse:
    try:
        return service.run_intervention(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model intervention failed: {exc}") from exc
