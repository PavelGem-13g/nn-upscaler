---
title: "Аудит открытой экосистемы и репозиториев (Open-Source Ecosystem Audit)"
tags: [research, open-source, github, huggingface, licenses, weights, reproducibility]
---

# Аудит открытой экосистемы и репозиториев (Open-Source Ecosystem Audit)

## 1. Краткое описание
В этой статье приводится детальный технический аудит проверенных открытых проектов, моделей Hugging Face и репозиториев GitHub, применимых для решения задачи Pixel Art → Realism. Для каждого проекта анализируются лицензионные ограничения, доступность предобученных весов, требования к оборудованию и сложность воспроизведения.

---

## 2. Систематическая таблица проверенных проектов

| Репозиторий / Модель | Базовая архитектура | Лицензия | Наличие весов | Обучающие скрипты | Минимальный GPU | Статус поддержки | Ссылка |
|---|---|---|---|---|---|---|---|
| **`xinsir/controlnet-tile-sdxl-1.0`** | SDXL 1.0 ControlNet | **Apache 2.0** | ✅ Safetensors (2.5 ГБ) | ✅ Открыты | 10 ГБ VRAM | Активен (2024) | [Hugging Face](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0) |
| **`jasperai/Flux.1-dev-Controlnet-Upscaler`** | FLUX.1-dev MM-DiT | Non-commercial | ✅ Safetensors (3.5 ГБ) | ❌ Закрыты | 16–24 ГБ | Активен (2025) | [Hugging Face](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) |
| **`XPixelGroup/DiffBIR`** | SD 1.5 / SDXL IR-CN | **Apache 2.0** | ✅ Открыты на HF | ✅ Полные | 12 ГБ VRAM | Активен | [GitHub](https://github.com/XPixelGroup/DiffBIR) |
| **`cswry/SeeSR`** | SD 2.1 base | **Apache 2.0** | ✅ Открыты на HF | ✅ Полные | 12 ГБ VRAM | Завершен (CVPR'24) | [GitHub](https://github.com/cswry/SeeSR) |
| **`Fanghua-Yu/SUPIR`** | SDXL base | Non-commercial | ✅ Открыты на HF | ❌ Только инференс | 24 ГБ VRAM | Активен (CVPR'24) | [GitHub](https://github.com/Fanghua-Yu/SUPIR) |
| **`cumulo-autumn/StreamDiffusion`** | LCM / SD-Turbo | **Apache 2.0** | ✅ Интеграция с HF | ✅ Полные | 8–12 ГБ | Активен | [GitHub](https://github.com/cumulo-autumn/StreamDiffusion) |
| **`sczhou/CodeFormer`** | VQ-GAN Prior | S-Lab License (NC) | ✅ Открыты на HF | ✅ Полные | 4 ГБ VRAM | Стабилен | [GitHub](https://github.com/sczhou/CodeFormer) |
| **`xinntao/Real-ESRGAN`** | RRDBNet | **BSD-3-Clause** | ✅ Открыты на GitHub | ✅ Полные | 2 ГБ VRAM | Стабилен | [GitHub](https://github.com/xinntao/Real-ESRGAN) |

---

## 3. Детальный разбор флагманских компонентов

### 1. `xinsir/controlnet-tile-sdxl-1.0`
- **Значение для проекта:** Фундаментальный краеугольный камень открытого пайплайна.
- **Особенности:** В отличие от официального адаптера SD 1.5 Tile от lllyasviel, модель обучена специально под высокое разрешение SDXL $1024\times 1024$. Отличается непревзойденной устойчивостью к шуму дизеринга и размытию.
- **Лицензия Apache 2.0** позволяет легально встраивать её в коммерческие игровые движки и студийные пайплайны.

### 2. `XPixelGroup/DiffBIR` (DiffBIR-v2)
- **Значение для проекта:** Двухстадийная архитектура (удаление деградации $\to$ генерация через диффузию).
- **Специфика:** Превосходно работает на фото, но требует калибровки Stage-1 модуля при встрече с ретро-дизерингом Bayer, иначе пытается размыть его как артефакт сжатия JPEG.

---

## 4. Лицензионные риски для коммерческих студий (Commercial Viability)

При выборе стека для коммерческого релиза игры необходимо учитывать юридические риски:
- **Безопасный коммерческий стек (Green Zone):**
  `SDXL Base 1.0` (OpenRAIL++) + `xinsir/controlnet-tile-sdxl-1.0` (Apache 2.0) + `Real-ESRGAN` (BSD-3).  
  *Все веса разрешены для монетизации и коммерческих релизов.*
- **Зона исследовательских ограничений (Research Only):**
  `FLUX.1-dev`, `SUPIR`, `CodeFormer` (S-Lab).  
  *Требуют коммерческого лицензирования у правообладателей.*

---

## 5. Связанные темы
- Обзор литературы: [`10_Research_and_Experiments/01_Literature_Review_and_State_of_the_Art.md`](01_Literature_Review_and_State_of_the_Art.md)
- Библиография: [`REFERENCES.md`](../REFERENCES.md)
- Итоговые рекомендации: [`11_Project_Roadmap/01_MVP_Specification_and_Implementation.md`](../11_Project_Roadmap/01_MVP_Specification_and_Implementation.md)
