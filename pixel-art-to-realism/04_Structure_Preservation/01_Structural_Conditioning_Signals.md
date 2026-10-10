---
title: "Структурные управляющие сигналы и подавление сеточных галлюцинаций (Grid Hallucination)"
tags: [structure-preservation, controlnet, canny, depth-anything, bilateral-filter, grid-hallucination]
---

# Структурные управляющие сигналы и подавление сеточных галлюцинаций (Grid Hallucination)

## 1. Краткое описание
В этой статье исследуется центральная инженерная проблема сохранения геометрии: предотвращение паразитного артефакта «сеточного гроба» (Grid Hallucination), возникающего при наивной подаче пиксельной графики в детекторы границ. Проводится сравнительный анализ управляющих модальностей (Tile, Canny, Lineart, Depth, Normal), математически обосновывается применение билатеральной фильтрации и формулируется оптимальный многоканальный протокол подготовки карт условий.

---

## 2. Интуитивное объяснение: Ловушка детектора Canny

Представьте, что вы отдали в строительную компанию чертеж дома, нарисованный в блокноте в клетку. Но неопытный архитектор воспринял линии тетрадной клетки как несущие железобетонные стены! В результате вместо просторной гостиной вам построили лабиринт из сотен крошечных каменных каморок размером $5\times 5$ сантиметров.

Ровно это происходит, когда пиксель-арт подаётся напрямую в стандартный детектор границ **Canny**:
1. Детектор Canny ищет локальные максимумы градиента яркости.
2. В пиксель-арте перепад яркости между соседними пикселями достигает максимума на каждой ступеньке сетки.
3. Canny обводит каждый квадратный пиксель черной каймой.
4. Диффузионная модель послушно генерирует реалистичный мир, аккуратно выкладывая его... кирпичной плиткой, кафелем или шахматными кубиками! Лицо персонажа покрывается швами плитки, превращаясь в монстра.

```
Входной спрайт            Наивный Canny детектор          Генерация диффузии
┌───┐                    ┌─┬─┐                           [ Кафельная плитка, ]
│   └───┐      ───>      │ ├─┤ (Сетка кубиков!)  ───>    [ мозаичный монстр, ]
└───┬───┘                └─┴─┘                           [ разрушенное лицо! ]
```

---

## 3. Сравнительный анализ модальностей управляющих сигналов

| Сигнал управления | Метод извлечения | Реакция на пиксельную сетку | Применимость к Pixel Art → Realism |
|---|---|---|---|
| **Canny Edges** | Двойная пороговая фильтрация градиентов Собеля | **Катастрофическая:** обводит каждый пиксель | ❌ Неприменимо без сильного размытия |
| **Lineart / SoftEdge** | PiDiNet / HED нейросети | Средняя: сглаживает диагонали, но оставляет волнистость | 🟡 Допустимо с предобработкой |
| **Monocular Depth** | **Depth Anything v2 Small** | **Отличная:** оценивает 3D-форму объекта, игнорируя шум | ✅ **Обязательный второй канал** |
| **Surface Normals** | DSINE / Normal Map Estimator | Хорошая: восстанавливает углы плоскостей | 🟢 Рекомендуется для хард-сюрфейс (броня) |
| **ControlNet-Tile** | Внутреннее размытие Гаусса + ресэмплинг | **Идеальная:** полностью инвариантен к шуму сетки | ✅ **Базовое структурное ядро** |

---

## 4. Математика предварительной билатеральной фильтрации (Anti-Aliasing Pre-Filtering)

Чтобы преобразовать дискретные пиксельные ступени в непрерывные анатомические линии, изображение перед извлечением контуров пропускается через билатеральный фильтр (Bilateral Filter).

В отличие от фильтра Гаусса (который размывает всё подряд, превращая лицо в кашу), билатеральный фильтр усредняет пиксели с весами, зависящими как от пространственного расстояния, так и от **разницы в цвете**:
$$I_{\text{smooth}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot g_{\sigma_s}(\|p - q\|) \cdot g_{\sigma_r}(\|I(p) - I(q)\|)$$
где:
$$g_{\sigma_s}(\|p - q\|) = \exp\left( -\frac{\|p - q\|^2}{2 \sigma_s^2} \right) \quad \text{(пространственное ядро)}$$
$$g_{\sigma_r}(\|I(p) - I(q)\|) = \exp\left( -\frac{\|I(p) - I(q)\|^2}{2 \sigma_r^2} \right) \quad \text{(цветовое фотометрическое ядро)}$$

#### Физический эффект:
- Внутри одного цветового кластера (на плече или плаще) пиксельные ступени полностью растворяются в единую гладкую поверхность.
- На внешней границе между плащом и фоном разница цветов $\|I(p) - I(q)\|$ велика, поэтому фотометрический вес стремится к нулю — **внешний силуэт персонажа сохраняет идеальную резкость!**

---

## 5. Практическая реализация: Модуль подготовки направляющих карт

```python
"""nn_upscaler.extractors.control_preprocessor: Preparation of structural conditioning maps."""

import cv2
import numpy as np
from PIL import Image


class StructuralConditioningExtractor:
    """Извлекатель сглаженных направляющих карт без артефактов блочной сетки."""

    def __init__(self, target_size: tuple[int, int] = (1024, 1024)):
        self.target_size = target_size

    def prepare_tile_condition(self, sprite: Image.Image) -> Image.Image:
        """Подготовка карты условий для ControlNet-Tile."""
        # 1. Апскейл Nearest сохраняет чистую цветовую палитру
        upscaled = sprite.resize(self.target_size, Image.Resampling.NEAREST)
        return upscaled

    def prepare_anti_aliased_edges(self, sprite: Image.Image) -> Image.Image:
        """Извлечение сглаженных контуров (Lineart/Canny) без сеточных артефактов."""
        upscaled = sprite.resize(self.target_size, Image.Resampling.NEAREST)
        arr = np.array(upscaled.convert("RGB"))

        # 2. Двусторонний билатеральный фильтр для подавления ступеней
        smoothed = cv2.bilateralFilter(arr, d=11, sigmaColor=85, sigmaSpace=85)

        # 3. Мягкий детектор границ Собеля
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.uint8(np.clip((magnitude / magnitude.max()) * 255, 0, 255))

        # Инвертируем: белые линии на чёрном фоне
        return Image.fromarray(magnitude)

    def prepare_depth_proxy(self, sprite: Image.Image) -> Image.Image:
        """Генерация псевдо-карты глубины на базе морфологического градиента яркости."""
        upscaled = sprite.resize(self.target_size, Image.Resampling.NEAREST)
        arr = np.array(upscaled.convert("RGB"))
        smoothed = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
        eq_depth = cv2.equalizeHist(gray)
        depth_rgb = cv2.cvtColor(eq_depth, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)
```

---

## 6. Исследовательский анализ

```
                  [ СРАВНЕНИЕ СТРАТЕГИЙ УПРАВЛЕНИЯ ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ Canny без фильтрации ]                          [ Tile + Bilateral Depth ]
- Сеточный гроб (Grid Hallucination)              - Плавная органическая форма
- Кафельная плитка на коже                        - Фотореалистичные поры и ткани
- Испорченный результат                           - SOTA коммерческое качество
```

**Рекомендованная связка (Golden Standard):**
1. **ControlNet-Tile** (вес $w=0.75$) на базе апскейленного `Nearest-Neighbor`.
2. **ControlNet-Depth** (вес $w=0.35$) на базе карты `Depth Anything v2`.
3. Полный отказ от жесткого Canny-детектора.

---

## 7. Связанные темы
- Управление вниманием: [`04_Structure_Preservation/02_Attention_Control_and_Feature_Injection.md`](02_Attention_Control_and_Feature_Injection.md)
- Стратегии ограничений: [`04_Structure_Preservation/05_Constraint_Level_Strategies.md`](05_Constraint_Level_Strategies.md)
- Архитектура ControlNet: [`03_Architectures/04_ControlNet_and_Adapters.md`](../03_Architectures/04_ControlNet_and_Adapters.md)
