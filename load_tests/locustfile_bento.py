import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")

_rng = random.Random(0)
_files = sorted(IMAGE_DIR.glob("*.jpg"))
if len(_files) < 50:
    raise RuntimeError(f"Expected at least 50 JPEG images, found {len(_files)}")
IMAGES = [(p.name, p.read_bytes()) for p in _rng.sample(_files, 50)]


class BentoPetUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task(20)
    def predict(self) -> None:
        name, data = random.choice(IMAGES)
        self.client.post(
            "/predict",
            files={"image": (name, data, "image/jpeg")},
            name="/predict",
        )

    @task(1)
    def health(self) -> None:
        self.client.post(
            "/health",
            json={},
            name="/health",
        )