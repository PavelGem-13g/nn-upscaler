# Neural Upscaler & Filter: Pixel Art to Realistic

Исследовательский проект по разработке нейросетевых моделей апскейлинга и фильтрации для превращения **пиксель-арта в детализированные, реалистичные (или околореалистичные) изображения**.

---

## 🎯 Цель проекта

- **Базовый уровень:** Качественный супер-разрешающий апскейлинг (Super-Resolution) без размытия и блочных артефактов (превосходящий Nearest-Neighbor, Bicubic, xBRZ).
- **План максимум:** Семантический депикселизатор (Depixelizer / Image-to-Image translation), который восстанавливает скрытую форму, текстуру, свет и детали персонажей/сцен из грубой низкополигональной сетки пикселей в околореалистичную картинку.

---

## 🧭 Документация для агентов и исследователей

- 📋 [**AGENTS.md**](AGENTS.md) — Протокол и правила для автономной работы AI-агентов.
- 🗺️ [**ROADMAP.md**](ROADMAP.md) — Фазы исследований от классических фильтров до генеративных моделей.
- 🔬 [**EXPERIMENTS.md**](EXPERIMENTS.md) — Журнал экспериментов, гипотез и метрик.

---

## ⚡ Быстрый старт

### 1. Установка окружения (uv)

Проект использует [`uv`](https://github.com/astral-sh/uv) для быстрого и воспроизводимого управления зависимостями:

```bash
# Клонирование и установка зависимостей
uv sync --extra dev
```

### 2. Запуск проверки окружения (Smoke Test)

Проверяет доступность MPS (Apple Silicon GPU), CUDA или CPU, инициализирует модели и делает тестовый прогон:

```bash
uv run python scripts/smoke_test.py
```

### 3. Тестовый апскейлинг изображения

```bash
# Классический апскейлинг 4x
uv run python scripts/upscale.py --input data/samples/sample_sprite.png --model nearest --scale 4 --output outputs/nearest_4x.png

# Нейросетевой апскейлинг 4x
uv run python scripts/upscale.py --input data/samples/sample_sprite.png --model edsr_light --scale 4 --output outputs/edsr_4x.png
```

### 4. Бенчмарк моделей

Сравнение всех доступных моделей по метрикам (PSNR, SSIM, время инференса):

```bash
uv run python scripts/benchmark.py --samples data/samples/
```

### 5. Экспорт и оптимизация моделей (ONNX, CoreML, TorchScript, Quantization)

```bash
# Экспорт модели в ONNX с динамическими осями и валидацией эквивалентности
uv run python scripts/export.py --model edsr_light --scale 4 --format onnx --verify

# Экспорт в половинную точность FP16 (экономия 50% памяти)
uv run python scripts/export.py --model edsr_light --scale 4 --format onnx --quantize fp16 --verify

# Экспорт для Apple Neural Engine / Apple Silicon (CoreML .mlpackage)
uv run python scripts/export.py --model edsr_light --scale 4 --format coreml --verify
```

### 6. Пакетная обработка спрайтов и текстур (Batch Processing CLI)

Высокопроизводительный инструмент для рекурсивного апскейлинга целых каталогов спрайтов и текстурных библиотек с сохранением прозрачности, прогресс-баром Rich и выгрузкой отчета `manifest.json`:

```bash
# Пакетный апскейлинг библиотеки спрайтов
uv run python scripts/batch_upscale.py \
    --input-dir data/benchmarks/authentic_pixel_art/ \
    --output-dir outputs/batch_sprites/ \
    --model edsr_light \
    --scale 4 \
    --recursive
```

### 7. Интерактивный Web UI (Gradio)

Современный браузерный интерфейс со сплит-слайдером сравнения «до/после», поддержкой всех архитектур, семантическим кондиционированием, фильтрацией дизеринга и телеметрией в реальном времени:

```bash
# Запуск локального веб-сервера (по умолчанию http://127.0.0.1:7860)
uv run python scripts/web_ui.py

# Запуск с публичной ссылкой (share)
uv run python scripts/web_ui.py --host 0.0.0.0 --port 7860 --share
```

### 8. Запуск тестов

```bash
uv run pytest -v
```

---

## 🏗 Архитектура репозитория

```text
nn-upscaler/
├── AGENTS.md                  # Инструкции и правила автономной работы для AI-агента
├── ROADMAP.md                 # Дорожная карта исследований
├── EXPERIMENTS.md             # Журнал экспериментов и результатов
├── pyproject.toml             # Зависимости и конфигурация проекта (uv)
├── configs/                   # Конфигурационные файлы экспериментов (YAML)
├── data/
│   ├── benchmarks/            # Аутентичные ретро-спрайты для валидации
│   ├── samples/               # Тестовые легковесные изображения для быстрой проверки
│   └── README.md              # Инструкция по загрузке и подготовке датасетов
├── src/nn_upscaler/
│   ├── export/                # Модули экспорта (ONNX, CoreML, Quantization PTQ)
│   ├── models/                # Архитектуры моделей (Classical, EDSR, RRDB, SwinIR, LatentDiffusion и др.)
│   ├── data/                  # Датасеты, синтетический пайплайн деградации (фото -> пиксель-арт)
│   ├── losses/                # Функции потерь (L1, Perceptual, RaGAN, PatchNCE, FFT)
│   ├── evaluation/            # Метрики (PSNR, SSIM, LPIPS) и бенчмарки
│   ├── inference/             # Пайплайн инференса (тайлинг, диффузия, адаптеры)
│   ├── training/              # Модульный цикл обучения с поддержкой MPS/CUDA
│   └── utils/                 # Определение устройств (MPS/CUDA/CPU), кэш, визуализация
├── scripts/                   # CLI-скрипты: export, batch_upscale, web_ui, train, upscale, benchmark, smoke_test
└── tests/                     # Модульные и интеграционные тесты
```

---

## 💻 Аппаратное ускорение

Код оптимизирован для работы на:
- **Apple Silicon (MPS & Apple Neural Engine)** — автоматический выбор `device="mps"` на Mac, нативный экспорт в CoreML `.mlpackage`.
- **Nvidia GPU (CUDA)** — при запуске на серверах / облаке с поддержкой TensorRT/ONNX.
- **CPU** — надёжный фолбэк для любых сред с поддержкой Dynamic INT8 Quantization.
