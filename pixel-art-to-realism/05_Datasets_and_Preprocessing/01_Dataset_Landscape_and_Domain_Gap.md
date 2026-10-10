---
title: "Ландшафт датасетов и исследование междоменного разрыва (Domain Gap)"
tags: [datasets, domain-gap, retro-sprites, spriters-resource, mmd, dinov2, data-curation]
---

# Ландшафт датасетов и исследование междоменного разрыва (Domain Gap)

## 1. Краткое описание
В этой статье проводится системный анализ открытых наборов данных для задачи Pixel Art → Realism. Исследуется фундаментальная проблема междоменного разрыва (Domain Gap) между настоящим авторским пиксель-артом художников и алгоритмически субдискретизированными фотографиями. Формулируется математический аппарат измерения расхождения распределений (Maximum Mean Discrepancy) и описываются стратегии гармонизации данных.

---

## 2. Интуитивное объяснение: Фотошоп против Художника

Представьте два изображения размером $32\times 32$:
1. **Первое изображение получено алгоритмически:** Взяли фотографию кота и уменьшили её в Photoshop через бикубическую интерполяцию, а затем применили автоматический дизеринг. На этом изображении каждый второй пиксель — одиночная случайная точка шума, усы кота превратились в пунктирную кашу, а глаза размазаны.
2. **Второе изображение нарисовано профессиональным пиксель-арт художником:** Каждый пиксель поставлен вручную. Усы кота обозначены непрерывной чистой линией толщиной ровно в 1 пиксель, зрачок — контрастной точкой, а шерсть сгруппирована в аккуратные монолитные световые пятна (кластеры).

Если обучить нейросеть только на первом (синтетическом) типе данных, при встрече с реальной игрой (второй тип) модель сойдёт с ума: она привыкла, что одиночные точки — это шум шерсти, и начнёт превращать чистые контуры спрайта в грязную рябь. Это расхождение называется **Domain Gap**.

---

## 3. Обзор доступных датасетов

```mermaid
flowchart TD
    subgraph PixelData["Непарный Pixel Art"]
        SR["The Spriters Resource<br/>> 500 000 игровых спрайтов<br/>(NES, SNES, GBA, Arcade)"]
        DB["Danbooru / Gelbooru<br/>~120 000 тегов 'pixel_art'<br/>(Современные арты художников)"]
    end

    subgraph RealData["Реалистичные фото высокого разрешения"]
        D2K["DIV2K / Flickr2K<br/>3 000 эталонных 2K/4K фото"]
        COCO["COCO-Stuff / OpenImages<br/>> 1.5 млн размеченных сцен"]
        LAION["LAION-Aesthetics (>6.5)<br/>Художественные концепты"]
    end

    subgraph Gap["МЕЖДОМЕННЫЙ РАЗРЫВ (DOMAIN GAP)"]
        SR -.-> Gap
        D2K -.-> Gap
    end

    Gap --> Solution["Решение: Псевдо-парный синтез + 3D рендеринг в Blender"]
```

### 3.1. Корпуса пиксельной графики
- **The Spriters Resource (TSR):** Крупнейший архив игровых спрайт-листов (NES, Sega Genesis, Neo-Geo, Game Boy Advance). Огромное разнообразие анимаций ходьбы, атак, оружия и тайлов. *Минусы:* Неразмеченные спрайт-листы, требующие автоматического нарезания на индивидуальные кадры по сетке.
- **Danbooru Pixel Art Collection:** Высокохудожественные законченные иллюстрации в изометрии и перспективе. *Плюсы:* Наличие детальных семантических текстовых тегов.

### 3.2. Корпуса реалистичных изображений
- **DIV2K / Flickr2K:** Золотой стандарт супер-разрешения. Идеальная резкость, отсутствие артефактов сжатия.
- **OpenImages v7:** Миллионы фото с ограничивающими рамками (bounding boxes) для обучения сегментации отдельных персонажей и оружия.

---

## 4. Математический анализ Domain Gap: Измерение через MMD

Для количественной оценки расхождения между распределением синтетического пиксель-арта $P$ и авторского ретро-пиксель-арта $Q$ используется критерий **Maximum Mean Discrepancy (MMD)** в пространстве признаков глубокой сети (DINOv2):
$$\operatorname{MMD}^2(\mathcal{F}, P, Q) = \mathbb{E}_{x, x' \sim P} [k(x, x')] - 2 \mathbb{E}_{x \sim P, y \sim Q} [k(x, y)] + \mathbb{E}_{y, y' \sim Q} [k(y, y')]$$
где $k(x, y) = \exp\left( -\frac{\|\phi(x) - \phi(y)\|^2}{2\sigma^2} \right)$ — радиально-базисное ядро (RBF) над нормализованными эмбеддингами DINOv2 $\phi(x)$.

```
   Распределение признаков DINOv2
      ▲
      │       Синтетический Pixel Art (P)          Настоящий ретро Pixel Art (Q)
Плот- │             ╭──────────╮                          ╭──────────╮
ность │            ╭╯          ╰╮                        ╭╯          ╰╮
      │           ╭╯            ╰╮                      ╭╯            ╰╮
      └───────────┴──────────────┴──────────────────────┴──────────────┴──────►
                                 ◄──── MMD Gap ────►
```

---

## 5. Практическая реализация: Расчет MMD расстояния между датасетами

```python
"""nn_upscaler.data.domain_gap: Maximum Mean Discrepancy (MMD) calculator using DINOv2 embeddings."""

import torch
import torch.nn as nn


class DomainGapCalculator:
    """Вычисление расхождения между синтетическим и авторским пиксель-артом."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        # Загрузка предобученной модели DINOv2
        self.encoder = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
        )

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        beta = 1.0 / (2.0 * sigma**2)
        dist_sq = torch.cdist(x, y) ** 2
        return torch.exp(-beta * dist_sq)

    @torch.no_grad()
    def calculate_mmd(self, dataset_p: torch.Tensor, dataset_q: torch.Tensor) -> float:
        """
        dataset_p: Батч синтетических спрайтов [N, 3, 224, 224]
        dataset_q: Батч реальных авторских спрайтов [M, 3, 224, 224]
        """
        # Извлечение глобальных эмбеддингов CLS токена
        feat_p = self.encoder(dataset_p.to(self.device))
        feat_q = self.encoder(dataset_q.to(self.device))

        # Нормализация признаков к единичной сфере
        feat_p = feat_p / feat_p.norm(dim=1, keepdim=True)
        feat_q = feat_q / feat_q.norm(dim=1, keepdim=True)

        k_pp = self.rbf_kernel(feat_p, feat_p).mean()
        k_qq = self.rbf_kernel(feat_q, feat_q).mean()
        k_pq = self.rbf_kernel(feat_p, feat_q).mean()

        mmd_sq = k_pp - 2.0 * k_pq + k_qq
        return float(torch.clamp(mmd_sq, min=0.0).sqrt().cpu().item())
```

---

## 6. Стратегии преодоления Domain Gap

1. **Морфологическая фильтрация сирот (Orphan Suppression):** Прогон синтетических пар через алгоритм очистки изолированных пикселей (см. [`01_Pixel_Art_Representation_and_Aesthetics.md`](../01_Fundamentals/01_Pixel_Art_Representation_and_Aesthetics.md)).
2. **Адаптация палитры (Palette Clamping):** Принудительная проекция синтетических изображений на реальные исторические палитры ретро-консолей (NES, PICO-8, GameBoy).
3. **Генерация 3D-пар (Unreal / Blender):** Использование 3D-движков для создания абсолютно точных пар «PBR-рендер $\leftrightarrow$ Pixel-art шейдер».

---

## 7. Связанные темы
- Синтетический пайплайн: [`05_Datasets_and_Preprocessing/02_Synthetic_Degradation_Pipeline.md`](02_Synthetic_Degradation_Pipeline.md)
- 3D синтез в Blender: [`05_Datasets_and_Preprocessing/03_3D_Engine_Synthetic_Data_Generation.md`](03_3D_Engine_Synthetic_Data_Generation.md)
- Фильтрация датасета: [`05_Datasets_and_Preprocessing/04_Dataset_Curation_and_Quality_Filtering.md`](04_Dataset_Curation_and_Quality_Filtering.md)
