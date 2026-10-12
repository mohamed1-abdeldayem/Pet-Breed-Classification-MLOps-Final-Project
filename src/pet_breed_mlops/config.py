from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="PET_", env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    model_dir: Path = Path("models")
    monitoring_baseline_path: Path = Path("reports/monitoring/baseline.json")
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
    threshold: float | None = None
    device: str = "cpu"