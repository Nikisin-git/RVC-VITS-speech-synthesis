# DEPLOYMENT.md — Развёртывание приложения синтеза и преобразования речи

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

### 5.1. Способ A — через Conda (рекомендуется)

#### Шаг 1. Установка системных зависимостей

**Windows:**
1. Установите [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.10, x64).
2. Установите [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) (последний studio/game ready).
3. Установите [CUDA Toolkit 11.8 или 12.1](https://developer.nvidia.com/cuda-toolkit-archive).
4. FFmpeg и espeak-ng доставляются portable-бинарниками в комплекте инсталлятора либо ставятся вручную:
   - FFmpeg: [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) → распаковать, добавить `bin/` в `PATH`.
   - espeak-ng: [GitHub releases](https://github.com/espeak-ng/espeak-ng/releases) → установить `.msi`.

**Ubuntu 22.04+:**
```bash
sudo apt update
sudo apt install -y ffmpeg espeak-ng build-essential
# NVIDIA-драйвер + CUDA Toolkit ставятся согласно официальной инструкции NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

#### Шаг 2. Клонирование репозитория

```bash
git clone https://github.com/Nikisin-git/RVC-VITS-speech-synthesis.git
cd RVC-VITS-speech-synthesis
```

#### Шаг 3. Создание conda-окружения

```bash
conda env create -f environment.yml
conda activate voicegen
```

Содержимое `environment.yml` (фрагмент):

```yaml
name: voicegen
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch=2.1.2
  - torchaudio=2.1.2
  - pytorch-cuda=11.8
  - ffmpeg=6.0
  - pip:
      - PySide6==6.6.3
      - demucs==4.0.1
      - deepfilternet==0.5.6
      - silero-vad
      - librosa
      - soundfile
      - pydub
      - pedalboard==0.9.12
      - openai-whisper
      - resemblyzer
      - transliterate
      - phonemizer
      - coqui-tts  # форк idiap/coqui-ai-TTS
      # RVC устанавливается из third_party/rvc_core
```

#### Шаг 4. Установка форка RVC

```bash
pip install -e third_party/rvc_core
```

#### Шаг 5. Запуск приложения

```bash
python -m app.main
```

При первом запуске будут скачаны предобученные модели (Demucs, DeepFilterNet, Whisper, RVC pretrained, VITS pretrained) в `user_data/cache/`.

---

### 5.2. Способ B — установщик-инсталлятор (Windows)

1. Скачайте `VoiceGen-Setup-<version>.exe` со страницы релизов.
2. Запустите от имени администратора.
3. Инсталлятор автоматически:
   - Установит Miniconda (если не установлена).
   - Создаст окружение `voicegen` из `environment.yml`.
   - Подложит FFmpeg как portable-бинарник в подпапку приложения (PATH не меняется глобально).
   - Установит espeak-ng.
   - При первом запуске скачает предобученные модели.
   - Создаст ярлык на рабочем столе.

Сборка инсталлятора локально: `installer/windows/inno_setup.iss` (Inno Setup 6+).

---

### 5.3. Способ C — Linux tar.gz архив

```bash
tar -xzf voicegen-<version>-linux-x64.tar.gz
cd voicegen-<version>
./install.sh        # вызывает installer/linux/install.sh
./voicegen          # запуск
```

Скрипт `install.sh` создаёт conda-окружение и проверяет наличие `ffmpeg` и `espeak-ng` в системе.

---

## 6. Проверка окружения после установки

При первом запуске приложение проверяет:
1. Наличие CUDA-совместимого GPU (`torch.cuda.is_available()`).
2. Версию драйвера NVIDIA и CUDA.
3. Наличие FFmpeg в PATH.
4. Свободное место на диске (минимум 5 ГБ).

Результаты доступны в окне «i» главного меню. При отсутствии GPU кнопки обучения RVC/VITS будут заблокированы.

Ручная проверка из консоли:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
ffmpeg -version
espeak-ng --version
```

---

## 7. Миграция на удалённый сервер (headless-режим)

Приложение спроектировано так, что ядро (`app/core/`) и CLI-скрипты (`scripts/`) работают независимо от Qt. Это позволяет запускать тяжёлые операции на сервере без GUI.

### 7.1. Подготовка сервера

Требования к серверу:
- Ubuntu 22.04+ x86_64.
- NVIDIA GPU + драйвер ≥ 525 + CUDA Toolkit 11.8 или 12.1.
- Docker (опционально) или нативно установленный conda.

### 7.2. Headless-запуск через CLI-скрипты

Все тяжёлые операции имеют отдельные CLI-обёртки в `scripts/`. Примеры:

```bash
# Отделение вокала
python scripts/run_demucs.py --input /data/audio.wav --out-dir /data/out --format wav

# Шумоподавление
python scripts/run_denoiser.py --input /data/audio.wav --out-dir /data/out --format wav

# Обучение RVC
python scripts/run_rvc_train.py \
    --dataset-dir /data/dataset \
    --model-name myvoice \
    --sample-rate 40k \
    --f0-method rmvpe_gpu \
    --epochs 200 \
    --save-every 10 \
    --batch-size 12

# Инференс RVC
python scripts/run_rvc_infer.py \
    --pth /models/myvoice.pth \
    --index /models/myvoice.index \
    --input /data/source.wav \
    --output /data/result.wav \
    --pitch 0 \
    --index-rate 0.5

# Обучение VITS
python scripts/run_vits_train.py \
    --audio-dir /data/dataset/wavs \
    --manifest /data/dataset/manifest.csv \
    --model-name mytts \
    --epochs 500 \
    --batch-size 16 \
    --mode finetune

# Инференс VITS
python scripts/run_vits_infer.py \
    --generator /models/mytts/G.pth \
    --config /models/mytts/config.json \
    --text "Привет, мир" \
    --output /data/tts_out.wav \
    --length-scale 1.0 \
    --pitch-shift 0
```

Все скрипты пишут структурированный лог в stdout и параллельно в `user_data/logs/`.

### 7.3. Docker-развёртывание

Базовый образ: `nvidia/cuda:11.8.0-runtime-ubuntu22.04`.

Пример `Dockerfile`:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    wget git build-essential ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app
COPY environment.yml /app/
RUN conda env create -f environment.yml && conda clean -afy

# Активация окружения по умолчанию
SHELL ["conda", "run", "-n", "voicegen", "/bin/bash", "-c"]

COPY . /app
RUN pip install -e third_party/rvc_core

ENV VOICEGEN_DATA_DIR=/data
ENV VOICEGEN_CACHE_DIR=/data/cache
VOLUME ["/data"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "voicegen", "python"]
CMD ["-m", "app.main", "--help"]
```

Запуск контейнера с GPU:
```bash
docker build -t voicegen:latest .
docker run --rm --gpus all \
    -v /host/data:/data \
    voicegen:latest \
    scripts/run_rvc_infer.py --pth /data/model.pth --input /data/in.wav --output /data/out.wav
```

### 7.4. Переменные окружения

| Переменная | Назначение | По умолчанию |
|------------|-----------|--------------|
| `VOICEGEN_DATA_DIR` | Корневой каталог `user_data/` | `<project>/user_data` |
| `VOICEGEN_CACHE_DIR` | Каталог кэша моделей | `$VOICEGEN_DATA_DIR/cache` |
| `VOICEGEN_LOG_LEVEL` | Уровень логирования | `INFO` |
| `CUDA_VISIBLE_DEVICES` | Выбор GPU | `0` |
| `PYTORCH_CUDA_ALLOC_CONF` | Тюнинг аллокатора PyTorch | — |

---

## 8. Подготовка датасета

### 8.1. RVC (преобразование голоса)
- 10 минут – 1 час чистой речи одного спикера.
- WAV, моно, 40 kHz или 48 kHz.
- Без музыки, без шумов (пропустить через блок предобработки).
- Нарезка фрагментами 5–20 секунд (предпочтительно по VAD).

### 8.2. VITS / TTS

**Режим «с нуля»:** 10–20 часов аудио высокого качества одного спикера.
**Режим «дообучения» (fine-tuning):** 10–30 минут.

Структура датасета:
```
dataset/
├── wavs/
│   ├── utt_001.wav
│   ├── utt_002.wav
│   └── ...
└── manifest.csv
```

Формат `manifest.csv` (LJSpeech, UTF-8, разделитель `|`, без заголовка):
```
utt_001.wav|Расшифровка первого фрагмента.
utt_002.wav|Расшифровка второго фрагмента.
```

Валидация перед обучением: все упомянутые файлы существуют, кодировка UTF-8, корректный разделитель, отсутствие осиротевших записей.

Скрипт автоматической подготовки манифеста: `scripts/build_tts_manifest.py` (обходит каталог `wavs/`, прогоняет каждый файл через Whisper для генерации первичной транскрипции, которую затем редактирует человек).

---

## 9. Решение типичных проблем

| Проблема | Причина / решение |
|----------|-------------------|
| `CUDA out of memory` | Уменьшить `batch_size`. Для VRAM 6 ГБ — 4, 8 ГБ — 8, 12 ГБ — 12+. |
| `torch.cuda.is_available() == False` | Проверить версии драйвера NVIDIA и CUDA Toolkit, переустановить PyTorch с правильным `pytorch-cuda`. |
| `ffmpeg: command not found` | Добавить FFmpeg в `PATH` или подложить portable-бинарник в подпапку приложения. |
| Битый чекпоинт после жёсткого прерывания | Чекпоинты пишутся атомарно через `.tmp` + `os.replace`. Загрузить предыдущий валидный из `TrainingModel/Assets/Weights/<имя_модели>/`. |
| Ошибки фонемизатора для русского | Установить `espeak-ng`, проверить `espeak-ng --version`. |
| Длинный путь в Windows (>260 симв.) | Имена результатов TTS усечены до 20 символов после транслитерации. Если путь всё равно длинный — переместить `user_data/` ближе к корню диска. |
| Зависший дочерний процесс | UI вызывает `QProcess.kill()` через кнопку «Отмена». Промежуточные файлы удаляются автоматически. |
| Долгое первое обучение | Это нормально: при первом запуске скачиваются предобученные модели (~3–5 ГБ суммарно). |

---

## 10. Сборка инсталлятора (для разработчиков)

### Windows (Inno Setup)
```bash
# Сборка conda-pack бандла
conda pack -n voicegen -o build/voicegen-env.tar.gz

# Сборка установщика
iscc installer/windows/inno_setup.iss
# Результат: installer/windows/Output/VoiceGen-Setup-<version>.exe
```

### Linux (tar.gz)
```bash
bash installer/linux/build_tarball.sh
# Результат: build/voicegen-<version>-linux-x64.tar.gz
```

---

## 11. Чек-лист готовности к ship'у

- [ ] `environment.yml` с зафиксированными версиями.
- [ ] Все CLI-скрипты в `scripts/` проверены на headless-запуск.
- [ ] Документация `README.md` + `DEPLOYMENT.md`.
- [ ] Инсталлятор Windows протестирован на чистой системе.
- [ ] Linux tar.gz протестирован на Ubuntu 22.04.
- [ ] Dockerfile собирается и запускается с `--gpus all`.
- [ ] Проверка окружения (`app/utils/env_check.py`) покрывает CUDA, FFmpeg, espeak-ng, свободное место.
- [ ] Тесты `tests/` проходят на CI.
