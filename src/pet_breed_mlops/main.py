from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import Settings
from .inference import PetBreedClassifier
from .preprocessing import load_image
from .schemas import BreedScore, HealthResponse, PredictionResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = PetBreedClassifier(
            cfg.model_dir, threshold=cfg.threshold, device=cfg.device
        )
        yield

    app = FastAPI(title="Pet Breed Classifier", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        clf: PetBreedClassifier = request.app.state.classifier
        return HealthResponse(
            status="healthy", model_version=clf.model_version, num_classes=clf.num_classes
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
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