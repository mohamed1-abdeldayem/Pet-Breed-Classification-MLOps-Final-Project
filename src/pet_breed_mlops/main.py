from __future__ import annotations

import json
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings
from .inference import PetBreedClassifier
from .monitoring import calculate_psi
from .preprocessing import load_image
from .schemas import BreedScore, HealthResponse, PredictionResponse

PREDICTION_REQUESTS = Counter(
    "pet_breed_prediction_requests_total",
    "Total prediction requests",
    ["status"],
)

PREDICTION_DURATION = Histogram(
    "pet_breed_prediction_duration_seconds",
    "Prediction request duration in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)

PREDICTION_CONFIDENCE = Histogram(
    "pet_breed_prediction_confidence",
    "Confidence scores for successful predictions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

DRIFT_SCORE = Gauge(
    "pet_breed_drift_score",
    "Latest measured data drift score",
)

DRIFT_WINDOW_READY = Gauge(
    "pet_breed_drift_window_ready",
    "1 when the drift observation window is full, otherwise 0",
)

class PredictionMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)

        try:
            response = await call_next(request)
        except Exception:
            PREDICTION_REQUESTS.labels(status="failure").inc()
            raise

        status = "success" if response.status_code < 400 else "failure"
        PREDICTION_REQUESTS.labels(status=status).inc()

        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = PetBreedClassifier(
            cfg.model_dir, threshold=cfg.threshold, device=cfg.device
        )
        yield

    app = FastAPI(title="Pet Breed Classifier", version="0.1.0", lifespan=lifespan)

    baseline_path = cfg.monitoring_baseline_path
    baseline_breeds: dict[str, int] = {}

    if baseline_path.is_file():
        baseline_data = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        baseline_breeds = baseline_data.get(
            "predicted_breed_distribution", {}
        )

    drift_window: deque[str] = deque(maxlen=100)
    DRIFT_WINDOW_READY.set(0)

    app.add_middleware(PredictionMetricsMiddleware)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        clf: PetBreedClassifier = request.app.state.classifier
        return HealthResponse(
            status="healthy", model_version=clf.model_version, num_classes=clf.num_classes
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        with PREDICTION_DURATION.time():

            clf: PetBreedClassifier = request.app.state.classifier

            data = await file.read(cfg.max_upload_bytes + 1)
            if len(data) > cfg.max_upload_bytes:
                raise HTTPException(422, f"File too large (max {cfg.max_upload_bytes} bytes)")
            if not data:
                raise HTTPException(422, "Empty file")

            try:
                image = load_image(data)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

            pred = await run_in_threadpool(clf.predict, image)
            PREDICTION_CONFIDENCE.observe(pred.confidence)

            drift_window.append(pred.breed)

            if baseline_breeds and len(drift_window) == drift_window.maxlen:
                drift_score = calculate_psi(baseline_breeds, drift_window)
                DRIFT_SCORE.set(drift_score)
                DRIFT_WINDOW_READY.set(1)

            return PredictionResponse(
                breed=pred.breed,
                species=pred.species,
                confidence=pred.confidence,
                top_3=[BreedScore(breed=b, probability=p) for b, p in pred.top_3],
                decision=pred.decision,
                model_version=pred.model_version,
            )

    return app


app = create_app()