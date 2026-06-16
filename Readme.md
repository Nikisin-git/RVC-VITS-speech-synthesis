# README.md — Сводная документация по приложению



Сводная документация по структуре проекта и развёртыванию настольного приложения для синтеза/преобразования речи (PySide6 + RVC + VITS).

---

## 1. Краткое описание

Настольное приложение на **PySide6** для:
- Предобработки аудиозаписей (отделение вокала, шумоподавление, удаление тишины, нарезка).
- Обучения и инференса моделей преобразования голоса (RVC).
- Обучения и инференса моделей синтеза речи (TTS на основе VITS).
- Оценки качества результата через метрики WER (Whisper) и SECS (resemblyzer).

**Поддерживаемые ОС:** Windows 10/11 (x64), Ubuntu 22.04+ (x86_64). macOS не поддерживается.

---

## 2. Минимальные системные требования

| Компонент | Минимум (только инференс) | Рекомендуется (обучение) |
|-----------|---------------------------|--------------------------|
| GPU | NVIDIA, 6 ГБ VRAM, CUDA Compute Capability ≥ 7.0 | NVIDIA RTX 3060/4060+, 12 ГБ VRAM |
| CUDA Toolkit | 11.8 или 12.1 | 12.1 |
| RAM | 16 ГБ | 32 ГБ |
| Диск (свободно) | 30 ГБ | 100 ГБ |
| CPU | 4 ядра, AVX2 | 8+ ядер |
| Python | 3.10.x (строго) | 3.10.x |
| ОС | Windows 10 21H2+ / Ubuntu 22.04+ | то же |

Системные зависимости (должны быть установлены отдельно или подложены portable-бинарниками):
- **FFmpeg 6.0+** (обязательно доступен в PATH).
- **espeak-ng** (нужен для русского TTS — фонемизация).
- **NVIDIA driver** + **CUDA Toolkit 11.8/12.1**.

---

## 3. Технологический стек

| Слой | Технология | Версия |
|------|------------|--------|
| GUI | PySide6 | 6.6.x |
| Аудио I/O | librosa, soundfile, pydub, ffmpeg-python | актуальные |
| Системные аудио-зависимости | FFmpeg | 6.0+ |
| Отделение вокала | Demucs (`htdemucs_ft`) | 4.0.1 |
| Шумоподавление | DeepFilterNet 3 | 0.5.6 |
| VAD (для нарезки) | silero-vad | 4.0+ |
| RVC | Форк ядра RVC-Project (CLI) | зафиксированный коммит |
| TTS | idiap/coqui-ai-TTS | 0.24+ |
| Постпроцессинг | pedalboard | 0.9+ |
| Метрики | openai-whisper (WER), resemblyzer (SECS) | актуальные |
| ML-фреймворк | PyTorch + torchaudio | 2.1.2 + CUDA 11.8/12.1 |
| Упаковка | PyInstaller / conda-pack | актуальные |

---

## 4. Структура проекта

```
voice-synthesis-app/
├── README.md
├── DEPLOYMENT.md                # Этот документ
├── environment.yml              # Conda-окружение с зафиксированными версиями
├── requirements.txt             # pip-зависимости (для pip-only установок)
├── pyproject.toml
├── LICENSE
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # Точка входа
│   ├── config.py                # Глобальные константы и пути
│   │
│   ├── ui/                      # Все Qt-окна и виджеты
│   │   ├── main_window.py
│   │   ├── info_dialog.py
│   │   ├── preprocessing/
│   │   │   ├── vocal_remover.py
│   │   │   ├── denoiser.py
│   │   │   ├── silence_remover.py
│   │   │   └── slicer.py
│   │   ├── rvc/
│   │   │   ├── train.py
│   │   │   └── inference.py
│   │   ├── tts/
│   │   │   ├── train.py
│   │   │   └── inference.py
│   │   └── widgets/             # Переиспользуемые виджеты
│   │       ├── drag_drop_area.py
│   │       ├── slider_with_input.py
│   │       ├── log_viewer.py
│   │       └── trapezoid_frame.py
│   │
│   ├── core/                    # Бизнес-логика (без UI)
│   │   ├── audio/
│   │   │   ├── vocal_separator.py
│   │   │   ├── denoiser.py
│   │   │   ├── silence_remover.py
│   │   │   └── slicer.py
│   │   ├── rvc/
│   │   │   ├── trainer.py
│   │   │   └── inferencer.py
│   │   ├── tts/
│   │   │   ├── trainer.py
│   │   │   └── inferencer.py
│   │   ├── metrics/
│   │   │   ├── wer.py
│   │   │   └── secs.py
│   │   └── postprocess/
│   │       └── pedalboard_chain.py
│   │
│   ├── workers/                 # QProcess-обёртки для subprocess'ов
│   │   ├── base_worker.py
│   │   ├── log_parser.py
│   │   └── progress_tracker.py
│   │
│   ├── styles/                  # QSS-темы
│   │   ├── dark.qss
│   │   ├── light.qss
│   │   ├── gray.qss
│   │   └── blue_gray.qss
│   │
│   ├── assets/
│   │   ├── icons/               # SVG-иконки
│   │   └── fonts/
│   │
│   └── utils/
│       ├── env_check.py         # Проверка GPU, CUDA, FFmpeg
│       ├── path_utils.py        # Защита от длинных путей в Windows
│       └── transliterate.py
│
├── scripts/                     # CLI-обёртки над core/ для QProcess
│   ├── run_demucs.py
│   ├── run_denoiser.py
│   ├── run_rvc_train.py
│   ├── run_rvc_infer.py
│   ├── run_vits_train.py
│   └── run_vits_infer.py
│
├── installer/
│   ├── windows/
│   │   ├── install.bat
│   │   └── inno_setup.iss       # Inno Setup script
│   └── linux/
│       └── install.sh
│
├── third_party/                 # Форки внешних библиотек
│   └── rvc_core/                # Извлечённое ядро RVC без Gradio
│
├── tests/
│
└── user_data/                   # Создаётся при первом запуске, в .gitignore
    ├── Audio Edit/
    │   ├── Vocal remover/
    │   ├── Denoising/
    │   ├── Unmuted/
    │   └── Cuts/
    ├── TrainingModel/
    ├── RVC models/
    ├── VITS/
    ├── Inference Results/
    ├── logs/
    └── cache/                   # Скачанные предобученные модели
```

**Принципы организации:**
- `app/ui/` — только Qt-виджеты и окна, никакой ML-логики.
- `app/core/` — чистая бизнес-логика, импортируется как из UI, так и из CLI-скриптов.
- `app/workers/` — обёртки `QProcess` для запуска тяжёлых операций в отдельных процессах.
- `scripts/` — CLI-точки входа для subprocess-запуска (через `QProcess` в GUI или вручную в headless-режиме).
- `user_data/` — все данные пользователя (аудио, чекпоинты, логи, кэш моделей). В `.gitignore`.

---

## 5. Локальная установка

См. раздел «Структура проекта» в [DEPLOYMENT.md](DEPLOYMENT.md).
