"""Global paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "VoiceGen"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

USER_DATA_DIR = Path(os.environ.get("VOICEGEN_DATA_DIR", PROJECT_ROOT / "user_data"))
CACHE_DIR = Path(os.environ.get("VOICEGEN_CACHE_DIR", USER_DATA_DIR / "cache"))
LOGS_DIR = USER_DATA_DIR / "logs"

AUDIO_EDIT_DIR = USER_DATA_DIR / "Audio Edit"
VOCAL_REMOVER_DIR = AUDIO_EDIT_DIR / "Vocal remover"
DENOISING_DIR = AUDIO_EDIT_DIR / "Denoising"
UNMUTED_DIR = AUDIO_EDIT_DIR / "Unmuted"
CUTS_DIR = AUDIO_EDIT_DIR / "Cuts"

TRAINING_DIR = USER_DATA_DIR / "TrainingModel"
RVC_MODELS_DIR = USER_DATA_DIR / "RVC models"
VITS_DIR = USER_DATA_DIR / "VITS"
INFERENCE_RESULTS_DIR = USER_DATA_DIR / "Inference Results"

DEFAULT_THEME = "dark"
AVAILABLE_THEMES = ("dark", "light", "gray", "blue_gray")

SUPPORTED_INPUT_FORMATS = (".wav", ".mp3")
SUPPORTED_OUTPUT_FORMATS = ("wav", "mp3")

MIN_FREE_DISK_GB = 5

F0_METHODS = {
    "pm": "Praat-based, быстрый, среднее качество. Для речи без сильных модуляций.",
    "harvest": "Высокая точность F0, медленный, на CPU. Подходит для пения.",
    "rmvpe": "Нейросетевой, точный, работает на CPU.",
    "rmvpe_gpu": "То же, но с ускорением на GPU. Рекомендуемый по умолчанию.",
}

SAMPLE_RATES = ("32k", "40k", "48k")


def ensure_dirs() -> None:
    """Create user_data subdirectories if absent."""
    for d in (
        USER_DATA_DIR,
        CACHE_DIR,
        LOGS_DIR,
        VOCAL_REMOVER_DIR / "vocals",
        VOCAL_REMOVER_DIR / "music",
        DENOISING_DIR,
        UNMUTED_DIR,
        CUTS_DIR / "Single track",
        CUTS_DIR / "Sliced",
        TRAINING_DIR,
        RVC_MODELS_DIR,
        VITS_DIR / "Assets" / "Weights",
        VITS_DIR / "Inference Results",
        INFERENCE_RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
