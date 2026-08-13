# Form Description.md — Разработанные экранные формы приложения

Документ описывает все GUI-формы (окна и диалоги) настольного приложения на PySide6.
Формы сгруппированы по функциональным блокам, соответствующим главному окну.

Модуль: `app/ui/`

---

## Содержание

- [1. Главное окно (`MainWindow`)](#1-главное-окно-mainwindow)
- [2. Диалог «О приложении» (`InfoDialog`)](#2-диалог-о-приложении-infodialog)
- [3. Формы предобработки аудио](#3-формы-предобработки-аудио)
  - [3.1. Отделение вокала (`VocalRemoverWindow`)](#31-отделение-вокала-vocalremoverwindow)
  - [3.2. Шумоподавление (`DenoiserWindow`)](#32-шумоподавление-denoiserwindow)
  - [3.3. Удаление тихих мест (`SilenceRemoverWindow`)](#33-удаление-тихих-мест-silenceremoverwindow)
  - [3.4. Нарезка на фрагменты (`SlicerWindow`)](#34-нарезка-на-фрагменты-slicerwindow)
- [4. Формы преобразования голоса (RVC)](#4-формы-преобразования-голоса-rvc)
  - [4.1. Обучение модели RVC (`RvcTrainWindow`)](#41-обучение-модели-rvc-rvctrainwindow)
  - [4.2. Инференс RVC (`RvcInferenceWindow`)](#42-инференс-rvc-rvcinferencewindow)
- [5. Формы синтеза речи (TTS/VITS)](#5-формы-синтеза-речи-ttsvits)
  - [5.1. Обучение TTS-модели (`TtsTrainWindow`)](#51-обучение-tts-модели-ttstrainwindow)
  - [5.2. Синтез речи (`TtsInferenceWindow`)](#52-синтез-речи-ttsinferencewindow)
- [6. Переиспользуемые виджеты](#6-переиспользуемые-виджеты)

---

## 1. Главное окно (`MainWindow`)

**Файл:** [`app/ui/main_window.py`](app/ui/main_window.py)

Точка входа в интерфейс. Содержит верхнюю панель (заголовок, переключатель темы,
кнопка «О приложении») и три функциональных блока в виде трапециевидных
рамок (`TrapezoidFrame`), соответствующих основным задачам приложения:

| Блок | Кнопки | Действие |
|------|--------|----------|
| Предобработка аудиозаписей | «Редактирование аудио» | открывает подменю с 4 формами предобработки |
| Преобразование голоса | «Обучить модель (RVC)», «Преобразовать голос» | открывает `RvcTrainWindow` / `RvcInferenceWindow` |
| Преобразование текста в речь | «Обучить модель (TTS)», «Преобразовать текст в речь» | открывает `TtsTrainWindow` / `TtsInferenceWindow` |

**Особенности:**
- Переключение QSS-темы через `QComboBox` (темы: `dark`, `light`, `gray`, `blue_gray`).
- В статус-баре — результат проверки окружения (`run_checks()`): CUDA, FFmpeg.
- При отсутствии CUDA кнопки обучения блокируются в соответствующих окнах.

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/01_main.png)

---

## 2. Диалог «О приложении» (`InfoDialog`)

**Файл:** [`app/ui/info_dialog.py`](app/ui/info_dialog.py)

Модальный диалог с информацией о приложении:
- Название и версия (`APP_NAME`, `APP_VERSION`).
- Блок «Системные требования» (минимум/рекомендация).
- Блок «Проверка окружения» — результаты `EnvReport.to_lines()` (CUDA, драйвер,
  версия PyTorch, наличие FFmpeg, объём VRAM).
- Ссылка на репозиторий с исходным кодом.

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/02_info_dialog.png)
---

## 3. Формы предобработки аудио

Все формы предобработки используют общий шаблон:
- Область drag-and-drop (`DragDropArea`) для .wav/.mp3.
- Радиокнопки выбора выходного формата (`make_format_radio`).
- Кнопка «Выполнить», запускающая CLI-скрипт через `ProcessWorker` (QProcess).
- Диалог прогресса (`ProgressDialog`) с логом и возможностью отмены.

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/03_audio.png)

  

### 3.1. Отделение вокала (`VocalRemoverWindow`)

**Файл:** [`app/ui/preprocessing/vocal_remover.py`](app/ui/preprocessing/vocal_remover.py)

Отделяет вокал от инструментала с помощью **Demucs** (`htdemucs_ft`).
Запускает `scripts/run_demucs.py` последовательно для каждого файла.

**Элементы формы:**
- Область drag-and-drop.
- Выбор формата вывода: WAV / MP3.
- Кнопка «Выполнить».

   ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/04_vocal_remover.png)

### 3.2. Шумоподавление (`DenoiserWindow`)

**Файл:** [`app/ui/preprocessing/denoiser.py`](app/ui/preprocessing/denoiser.py)

Убирает фоновый шум с помощью **DeepFilterNet 3**.
Запускает `scripts/run_denoiser.py`.

**Элементы формы:**
- Область drag-and-drop.
- Выбор формата вывода.
- Кнопка «Выполнить».

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/05_denoiser.png)

### 3.3. Удаление тихих мест (`SilenceRemoverWindow`)

**Файл:** [`app/ui/preprocessing/silence_remover.py`](app/ui/preprocessing/silence_remover.py)

Убирает паузы длиннее заданного порога. Запускает `scripts/run_silence_remover.py`.

**Элементы формы:**
- Область drag-and-drop.
- Выбор формата вывода.
- Слайдер «Допустимая длительность паузы, сек» (0.5–5.0, шаг 0.5).
- Слайдер «Порог тишины, дБ» (−100…0).
- Кнопка «Выполнить».

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/06_silence_remover.png)


### 3.4. Нарезка на фрагменты (`SlicerWindow`)

**Файл:** [`app/ui/preprocessing/slicer.py`](app/ui/preprocessing/slicer.py)

Режет аудио на равные фрагменты по таймеру или по VAD (silero-vad).
Запускает `scripts/run_slicer.py`.

**Элементы формы:**
- Область drag-and-drop.
- Выбор формата вывода.
- Группа «Сохранить аудиозаписи в виде»: единая дорожка / несколько фрагментов.
- Слайдер «Длительность фрагмента, сек» (5–20).
- Группа «Способ нарезки»: по таймеру / по тишине (VAD).
- Группа «Что делать с фрагментами короче 4 секунд»: сохранить / дописать к
  предыдущему / отбросить.
- Кнопка «Выполнить».

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/07_slicer.png)

---

## 4. Формы преобразования голоса (RVC)

### 4.1. Обучение модели RVC (`RvcTrainWindow`)

**Файл:** [`app/ui/rvc/train.py`](app/ui/rvc/train.py)

Запускает обучение RVC-модели через `scripts/run_rvc_train.py`.
Показывает живой график лоссов из tfevents.

**Элементы формы:**
- Drag-and-drop для файлов датасета (.wav/.mp3).
- Поле «Имя модели» с онлайн-валидацией (`validate_model_name`).
- Группа «Параметры обучения»:
  - `QComboBox` — частота дискретизации (40k, 48k, …).
  - `QComboBox` — метод извлечения F0 (`pm`, `harvest`, `rmvpe`, `rmvpe_gpu`) +
    контекстная подсказка под ним.
  - Слайдеры: «Частота сохранения чекпоинта» (5–50 эпох), «Количество эпох»
    (10–10000), `batch_size` (4–32).
  - Подсказка рекомендуемого `batch_size` по объёму VRAM.
  - `QCheckBox` «Создать .zip архив после обучения».
- Кнопка «Начать обучение».
- Двухступенчатая отмена: мягкая (по флагу — ждём конец эпохи), затем жёсткая (kill).

**Поведение при отсутствии CUDA:** кнопка «Начать обучение» отключена,
показывается предупреждение.

   ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/08_rvc_train.png)

### 4.2. Инференс RVC (`RvcInferenceWindow`)

**Файл:** [`app/ui/rvc/inference.py`](app/ui/rvc/inference.py)

Преобразование голоса с постпроцессингом (pedalboard) и метриками
(WER/SECS/MCD). Запускает `scripts/run_rvc_infer.py`.

**Элементы формы:**
- Две колонки:
  - «Модель»: drag-and-drop для `.pth` и `.index`.
  - «Входное аудио»: drag-and-drop для одного файла.
- Кнопка «Сброс всех параметров».
- Группа «Настройка преобразования голоса»:
  - Тон в полутонах (−12…12), Скорость индексации (0–1), Радиус фильтра (1–5),
    Скорость смешивания RMS (0–1), Скорость защиты (0–0.5).
- Группа «Настройка сведения аудио» — вкладки `QTabWidget`:
  - Ревербация: размер комнаты, влажность, сухость, демпфирование.
  - Компрессор: соотношение, порог, дБ.
  - Фильтры: HPF, LPF, лимитер.
  - Подавление шума (gate): порог, соотношение, атака, спад.
- Внизу: кнопка «Выполнить генерацию», выбор формата, аудиоплеер
  (`AudioPlayer`), надпись с метриками (WER / SECS / MCD).

     ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/09_rvc_inference.png)

---

## 5. Формы синтеза речи (TTS/VITS)

Синтез речи построен на архитектуре **VITS**. Приложение поддерживает **два
независимых движка обучения и инференса**, которые выбираются прямо в форме:

| Движок | Основа | Когда использовать |
|--------|--------|--------------------|
| **HuggingFace VITS** | дообучение готовой языковой базы Facebook **`mms-tts-rus`** (через `finetune-hf-vits`) | русский голос; достаточно 10–30 минут аудио одного диктора |
| **Coqui TTS VITS** | обучение с нуля или дообучение чекпойнта Coqui-TTS | англоязычные / собственные модели, полноразмерные датасеты |

Формат модели определяется автоматически по `config.json`
(`is_hf_vits_config()`): конфиг HuggingFace содержит
`"architectures": ["VitsModel"]`, конфиг Coqui — поле `"model"` и блоки
`"audio"`/`"model_args"`. Благодаря этому одна и та же форма синтеза работает с
обоими типами моделей.

### 5.1. Обучение TTS-модели (`TtsTrainWindow`)

**Файл:** [`app/ui/tts/train.py`](app/ui/tts/train.py)

Обучение / дообучение VITS-модели одним из двух движков. Переключатель движка
меняет набор видимых групп настроек. При первом открытии показывается
`QMessageBox` с рекомендацией (для русского голоса — «HuggingFace VITS»).

**Общие элементы формы:**
- Группа «Движок обучения»: радиокнопки «HuggingFace VITS (русский, mms-tts-rus)»
  (по умолчанию) / «Coqui VITS».
- Две колонки данных:
  - «Аудио» — поле выбора папки с `.wav` (`_FolderPicker` с кнопкой «Обзор…»,
    рекомендуется для больших датасетов до тысяч файлов) **или** drag-and-drop
    отдельных файлов для небольших наборов.
  - «Манифест» — drag-and-drop `.csv` в формате LJSpeech:
    `имя_файла.wav|расшифровка`.
- Группа «Настройки обучения»:
  - «Имя модели» с онлайн-валидацией (`validate_model_name`).
  - Слайдеры: «Частота сохранения чекпойнта, эпох» (5–50), «Количество эпох»
    (10–10000), `batch_size` (4–32, начальное значение — по объёму VRAM).
- Кнопка «Начать обучение».
- Перед запуском — валидация манифеста (`validate_manifest`): сверяет строки с
  наличием аудиофайлов в папке, критическое сообщение при ошибках.
- При отсутствии CUDA кнопка заблокирована, показывается предупреждение.
  
**Настройки движка «Coqui VITS»** (видны при выборе этого движка):
- Подгруппа «Режим обучения»: «С нуля (10–20 часов)» / «Дообучение (10–30 минут)»
  (по умолчанию — дообучение).
- Drag-and-drop базового чекпойнта `.pth` (активен и обязателен в режиме
  «Дообучение»; при отсутствии — предупреждение о риске расхождения обучения).
- Запуск через `scripts/run_vits_train.py`; строится живой график лоссов из
  tfevents, прогресс по эпохам.
- Двухступенчатая отмена: мягкая (по флагу — ждём конец эпохи), затем жёсткая (kill).
  
   ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/10_coqui_train.png)


**Настройки движка «HuggingFace VITS»** (видны при выборе этого движка):
- Поле «Репозиторий finetune-hf-vits» (`_FolderPicker`); по умолчанию берётся из
  переменной окружения `VOICEGEN_FINETUNE_HF_VITS` или из папки проекта.
- Поле «Базовая модель с дискриминатором (mms-tts-rus-with-disc)» (`_FolderPicker`).
- Подсказка о разовой подготовке (клон репозитория, сборка `monotonic_align`,
  подготовка базовой модели — см. `docs/HF_VITS_FINETUNE.md`).
- Запуск через `scripts/run_hf_vits_train.py`; результат — в
  `user_data/VITS/hf_vits_finetune/<имя>/run`. Живой график лоссов не строится
  (HF Trainer пишет собственные скаляры), прогресс виден по логу.

    ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/11_hugging_face_train.png)

  

### 5.2. Синтез речи (`TtsInferenceWindow`)

**Файл:** [`app/ui/tts/inference.py`](app/ui/tts/inference.py)

Синтез речи по тексту обученной VITS-моделью через
`scripts/run_vits_infer.py`. Тип модели (HuggingFace или Coqui) определяется
автоматически по `config.json`, поэтому форма единая для обоих движков.

**Элементы формы:**
- Группа «Модель»: две области drag-and-drop — `best_model.pth` и `config.json`.
  Для HuggingFace-модели `.pth` не требуется (веса лежат рядом с `config.json`
  как `model.safetensors`); для Coqui `.pth` обязателен.
- Кнопка «Сбросить параметры».
- Многострочный `QTextEdit` для текста синтеза.
- Группа «Параметры генерации»:
  - «Темп речи (меньше — медленнее)» — 0.25–2.0, шаг 0.05 (для HF задаёт
    `speaking_rate` модели, для Coqui — постобработку скорости).
  - «Высота голоса, полутонов» — −12…12 (сдвиг тона постобработкой через librosa).
- Внизу: кнопка «Выполнить генерацию», выбор формата сохранения (WAV / MP3),
  аудиоплеер (`AudioPlayer`) и строка метрик (WER / SECS / MCD).

   ![](https://github.com/Nikisin-git/RVC-VITS-speech-synthesis/blob/main/Applications/Forms/12_tts_inference.png)

---

## 6. Переиспользуемые виджеты

Модуль: [`app/ui/widgets/`](app/ui/widgets/)

| Виджет | Файл | Назначение |
|--------|------|------------|
| `DragDropArea` | `drag_drop_area.py` | Область drag-and-drop с фильтром расширений и режимом одиночного файла |
| `SliderWithInput` | `slider_with_input.py` | Слайдер с синхронизированным числовым полем и подписью |
| `ProgressDialog` | `progress_dialog.py` | Модальный диалог прогресса: лог, детерминированный/индетерминированный прогресс, живой график лоссов, кнопка отмены |
| `AudioPlayer` | `audio_player.py` | Плеер сгенерированного аудио (play/pause, перемотка, таймкод «текущее / общее»); проигрывает PCM напрямую через `QAudioSink` |
| `LogViewer` | `log_viewer.py` | Прокручиваемое поле для потокового лога |
| `TrainingChart` | `training_chart.py` | Живой график лоссов из `tfevents` (для RVC и TTS) |
| `TrapezoidFrame` | `trapezoid_frame.py` | Трапециевидная рамка для блоков главного окна |

Общие вспомогательные функции форм предобработки —
[`app/ui/preprocessing/_common.py`](app/ui/preprocessing/_common.py)
(например, `make_format_radio`).
