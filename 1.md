# VoiceGen — синтез и преобразование речи

Настольное приложение на PySide6 для:

- **Предобработки аудио** — отделение вокала (Demucs `htdemucs_ft`), шумоподавление (DeepFilterNet 3), удаление тишины, нарезка (по таймеру или VAD).
- **Преобразования голоса (RVC)** — обучение и инференс моделей RVC с пост-обработкой через pedalboard.
- **Синтеза речи (TTS)** — обучение и инференс VITS на базе `idiap/coqui-ai-TTS`, с фонемизацией через espeak-ng.
- **Оценки качества** — WER (Whisper) и SECS (resemblyzer).

## Быстрый старт

```bash
conda env create -f environment.yml
conda activate voicegen
pip install -e third_party/rvc_core   # форк ядра RVC без Gradio
python -m app.main
```

Подробные шаги, headless-режим, Docker, миграция на сервер — в [DEPLOYMENT.md](DEPLOYMENT.md).

## Системные требования

- ОС: Windows 10/11 (x64) или Ubuntu 22.04+. macOS не поддерживается.
- GPU: NVIDIA с 6+ ГБ VRAM, CUDA 11.8 или 12.1 (обучение требует GPU).
- Python: **строго 3.10.x**.
- Системные зависимости: FFmpeg 6.0+, espeak-ng (для русского TTS).

## Структура

См. раздел «Структура проекта» в [DEPLOYMENT.md](DEPLOYMENT.md).

## Headless-запуск

Каждая тяжёлая операция доступна как CLI-скрипт в `scripts/`:

```bash
python scripts/run_demucs.py --input track.wav --format wav
python scripts/run_rvc_infer.py --pth model.pth --index model.index --input src.wav --pitch 0
python scripts/run_vits_infer.py --generator G.pth --config config.json --text "Привет"
```


