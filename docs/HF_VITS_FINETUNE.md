# Дообучение VITS на русском (HuggingFace / mms-tts-rus)

Coqui-native русской VITS для дообучения не существует, поэтому русская
VITS-модель строится дообучением `facebook/mms-tts-rus` (HuggingFace
Transformers VITS) через тренировочный цикл
[`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits).
Результат — тоже VITS, инференс идёт через ту же форму «Преобразование
текста в речь» (формат определяется автоматически по `config.json`).

Почему так, а не «с нуля»: обучение VITS с нуля требует 10–20 часов речи.
На 30 минутах (≈367 клипов) оно расходится и даёт жужжание. Дообучение
сошедшейся модели на тех же 367 клипах даёт рабочий голос за 20–100 эпох.

## Разовая подготовка окружения

```bash
conda activate voicegen
pip install "datasets>=2.16" "accelerate>=0.26"

# 1. Клонировать тренировочный репозиторий
git clone https://github.com/ylacombe/finetune-hf-vits
cd finetune-hf-vits
pip install -r requirements.txt

# 2. Собрать monotonic_align (Cython)
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..

# 3. Прикрепить дискриминатор к базовой модели (MMS идёт без него).
#    Делается один раз для русского; результат — папка с базой,
#    пригодной для дообучения.
python convert_original_discriminator_checkpoint.py \
  --language_code rus \
  --pytorch_dump_folder_path ./mms-tts-rus-with-disc
```

Запомни два пути:
- путь к репозиторию `finetune-hf-vits` (например `C:\ml\finetune-hf-vits`);
- путь к базе с дискриминатором (`./mms-tts-rus-with-disc`).

## Дообучение на своём дикторе

Датасет — тот же, что для Coqui: папка с `.wav` + манифест `stem|text|text`
(нижний регистр, ASCII-пути, вне OneDrive).

```bash
set VOICEGEN_FINETUNE_HF_VITS=C:\ml\finetune-hf-vits

python scripts\run_hf_vits_train.py ^
  --manifest F:\Datasets\Andrey\metadata.csv ^
  --audio-dir F:\Datasets\Andrey\wavs ^
  --model-name Andrey_HF ^
  --base-model .\mms-tts-rus-with-disc ^
  --output-dir F:\hf_vits_finetune\Andrey ^
  --epochs 100 ^
  --batch-size 8
```

Что делает скрипт:
1. Конвертирует манифест+wav в формат `audiofolder`
   (`scripts\prepare_hf_vits_dataset.py`).
2. Пишет train-конфиг с весами лоссов из эталонного примера
   (`weight_mel=35`, `weight_disc=3` и т.д.) и `learning_rate=2e-5`.
3. На картах без Tensor Cores (GTX 16xx) выключает fp16 — тот же фикс,
   что убрал расхождение в Coqui.
4. Монки-патчит `datasets.load_dataset`, чтобы `audiofolder` брал наш
   локальный `data_dir` (апстрим локальные пути напрямую не поддерживает).
5. Запускает `run_vits_finetuning.py`.

На GTX 1660 — ориентировочно 1–3 часа на 100 эпох. Выход: папка
`F:\hf_vits_finetune\Andrey\run` с `config.json` + `model.safetensors`.

## Инференс

В форме «Преобразование текста в речь»:
- в поле **config.json** укажи `...\run\config.json` из результата
  дообучения (или `config.json` от `facebook/mms-tts-rus`, чтобы
  проверить базовую модель без дообучения);
- поле **best_model.pth** оставь пустым — для HF-VITS веса лежат рядом с
  `config.json` (`model.safetensors`), отдельный `.pth` не нужен;
- введи текст, нажми «Выполнить генерацию».

Формат определяется автоматически: если в `config.json` есть
`"architectures": ["VitsModel"]` — идёт HF-путь, иначе Coqui.

Метрики WER / SECS / MCD считаются так же; для SECS/MCD положи
`reference_speaker.wav` (любой клип диктора) рядом с `config.json`.

## Замечания

- **Частота 16 кГц** — mms-tts-rus работает на 16 кГц. Для речи нормально,
  просто не студийные 22 кГц.
- **uroman** — токенайзер некоторых MMS-языков требует романизации. Для
  русского это обычно не нужно; если тренер попросит `uroman`, поставь его
  по инструкции из finetune-hf-vits.
- **monotonic_align на Windows** — если сборка Cython падает, поставь
  Build Tools for Visual Studio (C++), затем повтори `build_ext`.
