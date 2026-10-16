import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_DIR = Path("data/raw/oxford-iiit-pet/images")

# نحمّل 50 صورة ثابتة (seed ثابت) في الذاكرة مرة واحدة، فالقياس مش بيشمل قراءة الـ disk
_rng = random.Random(0)
_files = sorted(IMAGE_DIR.glob("*.jpg"))
IMAGES = [(p.name, p.read_bytes()) for p in _rng.sample(_files, 50)]


class PetUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task(20)
    def predict(self) -> None:
        name, data = random.choice(IMAGES)
        self.client.post(
            "/predict", files={"file": (name, data, "image/jpeg")}, name="/predict"
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")