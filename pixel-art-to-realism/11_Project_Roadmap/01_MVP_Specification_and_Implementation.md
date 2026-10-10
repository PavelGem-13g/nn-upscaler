---
title: "Спецификация и пошаговая реализация MVP (Minimum Viable Product)"
tags: [roadmap, mvp, specification, acceptance-criteria, sdxl, controlnet-tile, quickstart]
---

# Спецификация и пошаговая реализация MVP (Minimum Viable Product)

## 1. Краткое описание
В этой статье формулируется законченная инженерная спецификация первого работающего прототипа (MVP) системы **Pixel Art → Realism** в репозитории `nn-upscaler`. Описываются точные версии весов, конфигурации планировщика, аппаратные требования, критерии приёмки (Acceptance Criteria) и пошаговый план тестирования.

---

## 2. Архитектурная формула MVP

Для минимизации рисков и немедленного получения коммерческого качества в первом прототипе выбран **Zero-Shot стек с аппаратным ускорением**, не требующий предварительного обучения с нуля:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ФОРМУЛА ЭТАЛОННОГО MVP                          │
│                                                                        │
│   Базовое ядро:       Stable Diffusion XL Base 1.0 (FP16/BF16)         │
│   Структурный якорь:  xinsir/controlnet-tile-sdxl-1.0 (Вес = 0.75)     │
│   Предобработка:      Nearest-Neighbor 1024x1024 + Bilateral Filter    │
│   Планировщик:        DPM++ 2M Karras (28 шагов, CFG = 7.0)            │
│   Постобработка:      Reinhard Lab Color Transfer                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Полная техническая спецификация MVP

### 3.1. Используемые компоненты и модели
- **Base Checkpoint:** `stabilityai/stable-diffusion-xl-base-1.0` (OpenRAIL++).
- **VAE Patch:** `madebyollin/sdxl-vae-fp16-fix` (устраняет черные экраны на GPU).
- **ControlNet:** `xinsir/controlnet-tile-sdxl-1.0` (Apache 2.0).
- **Scheduler:** `DPMSolverMultistepScheduler` (algorithm_type="dpmsolver++", use_karras_sigmas=True).

### 3.2. Гиперпараметры инференса
- **Целевое разрешение:** $1024 \times 1024$ пикселей.
- **Шаги сэмплирования:** 28 шагов.
- **Guidance Scale (CFG):** 7.0.
- **ControlNet Conditioning Scale:** 0.75.
- **ControlNet End Step:** 0.85 (освобождает последние 15% шагов для генерации микротекстур).
- **Параметры билатерального фильтра:** `d = 9`, `sigmaColor = 75`, `sigmaSpace = 75`.

---

## 4. Пошаговый запуск и воспроизведение MVP

Все зависимости и код уже интегрированы в репозиторий `nn-upscaler`:

```bash
# 1. Синхронизация зависимостей окружения через uv
uv sync --extra dev

# 2. Запуск санитарной проверки окружения и доступности GPU (MPS / CUDA)
uv run python scripts/smoke_test.py

# 3. Запуск инференса MVP на тестовом спрайте
uv run python scripts/upscale_diffusion.py \
  --input data/samples/sample_sprite.png \
  --output outputs/mvp_result.png \
  --prompt "photorealistic highly detailed knight portrait, polished steel plate armor, cinematic lighting, 8k resolution, raw photograph" \
  --tile-weight 0.75 \
  --steps 28 \
  --seed 42
```

---

## 5. Критерии приемки MVP (Acceptance Criteria)

Прототип считается успешным и готовым к переходу на Фазу 2 тогда и только тогда, когда выполняются **все** следующие условия:

1. **Пиковое потребление VRAM:** $\le 11.5$ ГБ на CUDA (помещается в RTX 3060 / 4060 Ti) и $\le 16$ ГБ Unified Memory на Apple Silicon Mac.
2. **Время инференса:** $\le 8.5$ секунд на NVIDIA RTX 4080/4090; $\le 25$ секунд на Apple Silicon M-series.
3. **Семантическая стабильность:** Семантическое сходство DINOv2 Cosine Similarity между входом и выходом $\ge 0.75$ (отсутствие мутаций классов объектов).
4. **Структурная стабильность:** Коэффициент перекрытия силуэта SAM Mask $IoU \ge 0.82$.
5. **Визуальная чистота:** Полное отсутствие артефакта «сеточного гроба» (квадратных кафельных плиток на коже и одежде).

---

## 6. Связанные темы
- Исходный CLI-скрипт: [`scripts/upscale_diffusion.py`](../../scripts/upscale_diffusion.py)
- Дорожная карта экспериментов: [`11_Project_Roadmap/02_Experiment_Roadmap_and_Decision_Gates.md`](02_Experiment_Roadmap_and_Decision_Gates.md)
- Анализ ошибок: [`08_Practical_Implementation/05_Diagnostics_and_Common_Bugs.md`](../08_Practical_Implementation/05_Diagnostics_and_Common_Bugs.md)
