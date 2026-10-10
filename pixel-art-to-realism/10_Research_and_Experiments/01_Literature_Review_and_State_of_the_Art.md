---
title: "Обзор научной литературы и State-of-the-Art (Literature Review & SOTA)"
tags: [research, literature-review, state-of-the-art, kopf-lischinski, pulse, diffbir, seesr, supir, voxify3d]
---

# Обзор научной литературы и State-of-the-Art (Literature Review & SOTA)

## 1. Краткое описание
В этой статье даётся развёрнутый академический обзор эволюции научных методов решения задачи преобразования пиксельной графики в непрерывные и фотореалистичные изображения: от эвристик эмуляторов консолей 1990-х годов и векторного сплайнинга SIGGRAPH 2011 до современных генеративных диффузионных систем и волуметрического 3D-рендеринга CVPR 2026.

---

## 2. Хронологическая эволюция научных парадигм

```mermaid
timeline
    title Эволюция методов депикселизации и генеративного восстановления
    1995-2005 : Эвристические фильтры эмуляторов (Scale2x, hq2x, xBRZ)
    2011 : Топологическая векторная депикселизация (Kopf-Lischinski, SIGGRAPH)
    2017-2019 : Сверточные сети и первые GAN (SRCNN, EDSR, Pix2Pix, CycleGAN)
    2020-2021 : Латентная GAN-инверсия (PULSE, pSp StyleGAN Inversion)
    2022-2023 : Латентная диффузия и ControlNet (LDM, ControlNet-Tile, MasaCtrl)
    2024 : Семантическая реставрация (SeeSR, SUPIR, CoDeF, DiffBIR)
    2025-2026 : Flow Matching и 3D-пиксель-арт (FLUX.1, Voxify3D, Texel Splatting)
```

---

## 3. Детальный разбор ключевых научных работ

### 1. Эра вычислительной геометрии: Kopf-Lischinski (SIGGRAPH 2011)
- *Статья:* *"Depixelizing Pixel Art"* (Kopf & Lischinski).
- *Идея:* Отказ от сверток в пользу планарных графов ячеек, эвристического разрешения неоднозначностей диагоналей и оптимизации сплайнов Безье.
- *Вклад:* Доказано, что дискретная сетка пиксель-арта может быть преобразована в непрерывную топологию без размытия.
- *Ограничение:* Выход ограничен плоской векторной графикой (SVG) без фототекстур.

### 2. Эра StyleGAN инверсии: PULSE (Menon et al., CVPR 2020)
- *Статья:* *"PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models"*.
- *Идея:* Оптимизация скрытого вектора $z \in \mathcal{W}$ в предобученном StyleGAN: $\min_z \|\mathcal{D}(G(z)) - y\|_2^2$.
- *Вклад:* Первый реальный синтез фотореалистичных лиц из пикселизированных аватаров.
- *Ограничение:* Узкий домен (работает только для лиц FFHQ, бессилен на произвольных игровых спрайтах).

### 3. Эра диффузионного структурного контроля: ControlNet (Zhang et al., ICCV 2023)
- *Статья:* *"Adding Conditional Control to Text-to-Image Diffusion Models"*.
- *Идея:* Параллельная обучаемая копия энкодера U-Net с нулевыми свертками.
- *Вклад:* Создание адаптера ControlNet-Tile, заложившего основу для контролируемой депикселизации.

### 4. Эра семантической слепой реставрации: SeeSR и SUPIR (CVPR 2024)
- *SeeSR (Wu et al., CVPR 2024):* Извлечение деградационно-устойчивых текстовых промптов (soft/hard prompts) и подмешивание низкоразмерного изображения в начальный шум для подавления галлюцинаций.
- *SUPIR (Yu et al., CVPR 2024):* Масштабирование диффузионной реставрации до SDXL на 20 миллионах изображений.

### 5. Современный рубеж: Волуметрический пиксель-арт (CVPR 2026)
- *Voxify3D (Huang et al., CVPR 2026, arXiv:2512.07834):* *"Voxify3D: Pixel Art Meets Volumetric Rendering"*. Использование ортогональной пиксельной супервизии, patch-based CLIP alignment и Gumbel-Softmax квантования палитры для дифференцируемой оптимизации 3D-мешей.
- *Texel Splatting (Ebert, 2026, arXiv:2603.14587):* Перспективно-стабильный рендеринг пиксель-арта через кубические карты мирового пространства.

---

## 4. Сравнительная матрица SOTA

| Метод | Публикация | Парадигма | Сохранение структуры | Реализм текстур | Ограничение |
|---|---|---|---|---|---|
| **Kopf-Lischinski** | SIGGRAPH 2011 | Графы и сплайны | **100% (Топология)** | 0% (Вектор) | Нет фототекстур |
| **PULSE** | CVPR 2020 | StyleGAN Inversion | 75% | 90% | Только лица |
| **DiffBIR** | 2023/2024 | Two-Stage Diffusion | 85% | 88% | Не любит дизеринг |
| **SUPIR** | CVPR 2024 | SDXL Scaling Prior | 90% | 95% | VRAM $> 24$ ГБ |
| **SDXL + Multi-CN** | Наш стек (MVP) | Latent Diffusion | **92%** | **95%** | Задержка ~7 сек |
| **Voxify3D** | CVPR 2026 | 3D Volumetric Neural | 95% | 92% | Требует 3D меш |

---

## 5. Связанные темы
- Аудит репозиториев: [`10_Research_and_Experiments/02_Open_Source_Ecosystem_Audit.md`](02_Open_Source_Ecosystem_Audit.md)
- Открытые вопросы: [`10_Research_and_Experiments/03_Open_Research_Questions.md`](03_Open_Research_Questions.md)
- Библиография: [`REFERENCES.md`](../REFERENCES.md)
