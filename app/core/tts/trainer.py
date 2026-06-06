"""VITS / Coqui-TTS training orchestration.

Uses `idiap/coqui-ai-TTS` (installed as `coqui-tts`). The Trainer is run as a
subprocess from `scripts/run_vits_train.py` so the parent UI can kill it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.config import VITS_DIR


class TtsTrainMode(str, Enum):
    SCRATCH = "scratch"
    FINETUNE = "finetune"


@dataclass
class TtsTrainConfig:
    audio_dir: Path
    manifest_path: Path
    model_name: str
    mode: TtsTrainMode = TtsTrainMode.FINETUNE
    epochs: int = 500
    save_every: int = 10
    batch_size: int = 16
    language: str = "ru"
    sample_rate: int = 22050
    pretrained_checkpoint: Path | None = None


def model_dir(model_name: str) -> Path:
    d = VITS_DIR / "Assets" / "Weights" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def build_coqui_config(cfg: TtsTrainConfig) -> dict:
    """Build a Coqui-TTS VITS config dict serializable to JSON."""
    out = {
        "model": "vits",
        "run_name": cfg.model_name,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "eval_batch_size": max(2, cfg.batch_size // 2),
        "save_step": 1000,
        "save_n_checkpoints": 5,
        "save_checkpoints": True,
        "print_step": 25,
        "print_eval": True,
        "mixed_precision": True,
        "output_path": str(model_dir(cfg.model_name)),
        "datasets": [{
            "formatter": "ljspeech",
            "dataset_name": cfg.model_name,
            "path": str(cfg.audio_dir.parent),
            "meta_file_train": cfg.manifest_path.name,
            "language": cfg.language,
        }],
        "audio": {
            "sample_rate": cfg.sample_rate,
        },
        "characters": {
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>",
            "characters_class": "TTS.tts.utils.text.characters.IPAPhonemes",
        },
        "phonemizer": "espeak",
        "phoneme_language": cfg.language,
        "use_phonemes": True,
        "test_sentences": [],
    }
    if cfg.mode == TtsTrainMode.FINETUNE and cfg.pretrained_checkpoint:
        out["restore_path"] = str(cfg.pretrained_checkpoint)
    return out


def save_config(cfg: TtsTrainConfig) -> Path:
    config = build_coqui_config(cfg)
    target = model_dir(cfg.model_name) / "config.json"
    _atomic_write(target, json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8"))
    return target


def _validate_manifest(cfg: TtsTrainConfig) -> None:
    """Catch typical mistakes in the manifest before Coqui chokes on them.

    Expected LJSpeech format: `audio_filename|text` or
    `audio_filename|text|normalized_text`, where the filename is RELATIVE
    to audio_dir. Absolute paths (especially Windows E:/... from someone
    else's dataset) and missing files are the most common failure mode.
    """
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Манифест не найден: {cfg.manifest_path}")

    lines = [ln.strip() for ln in cfg.manifest_path.read_text(
        encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Манифест пуст.")

    missing: list[str] = []
    abs_paths = 0
    for ln in lines[:50]:  # sample the first 50 rows
        parts = ln.split("|")
        if len(parts) < 2:
            raise ValueError(
                f"Строка манифеста не соответствует формату 'путь|текст': {ln[:80]!r}"
            )
        rel = parts[0]
        p = Path(rel)
        if p.is_absolute() or (len(rel) >= 2 and rel[1] == ":"):
            abs_paths += 1
            candidate = cfg.audio_dir / Path(rel).name
        else:
            candidate = cfg.audio_dir / rel
        # Coqui ljspeech appends '.wav' itself, so the manifest may store
        # bare stems. Try both.
        if not candidate.exists() and not str(candidate).lower().endswith(".wav"):
            candidate = candidate.with_name(candidate.name + ".wav")
        if not candidate.exists():
            missing.append(rel)

    if abs_paths:
        print(
            f"WARN: {abs_paths}/{min(50, len(lines))} строк используют абсолютные пути. "
            "Coqui-TTS ждёт пути относительно audio_dir; пробуем искать по имени файла.",
            flush=True,
        )
    if missing:
        sample = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            f"Не найдено {len(missing)} аудиофайлов из манифеста в {cfg.audio_dir}. "
            f"Примеры:\n  {sample}\n"
            "Проверьте, что аудио лежит в audio_dir и пути в манифесте указывают "
            "на реальные имена файлов."
        )


def train(cfg: TtsTrainConfig, cancel_flag: Path | None = None) -> dict:
    """Launch Coqui Trainer in-process.

    For UI-driven cancellation, this function is invoked from a subprocess script;
    the parent kills the subprocess. `cancel_flag` is checked in a callback hook.
    """
    _validate_manifest(cfg)

    try:
        from trainer import Trainer, TrainerArgs  # type: ignore  # provided by coqui-tts
        from TTS.config import load_config  # type: ignore
        from TTS.tts.configs.vits_config import VitsConfig  # type: ignore
        from TTS.tts.datasets import load_tts_samples  # type: ignore
        from TTS.tts.models.vits import Vits  # type: ignore
        from TTS.tts.utils.text.tokenizer import TTSTokenizer  # type: ignore
        from TTS.utils.audio import AudioProcessor  # type: ignore
    except Exception as e:
        # coqui-tts pulls transformers; on transformers 4.50+ the lazy module
        # system breaks with "Could not import module 'GenerationMixin'".
        raise RuntimeError(
            f"Не удалось импортировать Coqui-TTS ({type(e).__name__}: {e}). "
            "Чаще всего это несовместимая версия transformers — установите "
            "'pip install \"transformers>=4.57,<5.0\"' и попробуйте снова."
        ) from e

    config_path = save_config(cfg)
    config = VitsConfig()
    config.load_json(str(config_path))

    ap = AudioProcessor.init_from_config(config)
    # The tokenizer must be initialized before the model, otherwise
    # VitsDataset.__init__ crashes with `'NoneType' has no attribute 'use_phonemes'`.
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        config.datasets, eval_split=True, eval_split_size=0.05,
    )
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer_args = TrainerArgs(
        restore_path=str(cfg.pretrained_checkpoint) if cfg.pretrained_checkpoint else "",
    )
    trainer = Trainer(
        trainer_args, config, output_path=config.output_path,
        model=model, train_samples=train_samples, eval_samples=eval_samples,
    )

    if cancel_flag is not None:
        original_on_epoch_end = getattr(trainer, "on_epoch_end", None)

        def _wrapped(*a, **kw):
            if cancel_flag.exists():
                print("[tts] cancel flag detected — stopping after current epoch", flush=True)
                trainer.keep_running = False
            if original_on_epoch_end:
                return original_on_epoch_end(*a, **kw)

        trainer.on_epoch_end = _wrapped  # type: ignore

    trainer.fit()

    weights_dir = model_dir(cfg.model_name)
    return {
        "weights_dir": str(weights_dir),
        "config": str(config_path),
        "generator": str(weights_dir / "G.pth"),
        "discriminator": str(weights_dir / "D.pth"),
    }
