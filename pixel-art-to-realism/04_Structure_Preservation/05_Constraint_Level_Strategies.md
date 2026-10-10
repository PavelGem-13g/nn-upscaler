---
title: "Стратегии уровней ограничений (Constraint Level Strategies)"
tags: [structure-preservation, constraint-levels, inference-presets, hyperparameters, controlnet-weights]
---

# Стратегии уровней ограничений (Constraint Level Strategies)

## 1. Краткое описание
В этой статье формулируются три стандартизированные операционные стратегии управления преобразованием: **Strict Structural Anchor** (строгая блочная фиксация), **Balanced Semantic Translation** (сбалансированная органическая трансформация) и **Free Creative Hallucination** (свободная художественная реинтерпретация). Приводятся точные таблицы гиперпараметров, описывается явление фазового перехода геометрии и даётся практический конфигуратор пресетов.

---

## 2. Интуитивное объяснение: Регулятор свободы воображения

В реальной разработке видеоигр требования к результату радикально различаются в зависимости от типа ассета:
- Когда геймдизайнер даёт вам **тайл каменного пола $16\times 16$**, он требует: *«Не смей сдвигать швы между камнями ни на один миллиметр, иначе соседние тайлы не состыкуются!»* (Уровень свободы = 10%).
- Когда художник отдаёт **спрайт рыцаря $32\times 32$**, он говорит: *«Сделай его живым человеком! Сгладь угловатые плечи в анатомические мышцы, преврати серые квадраты в полированную сталь, но сохрани позу, цвет плаща и форму меча»* (Уровень свободы = 50%).
- Когда арт-директор просит **постер по мотивам уровня игры**, он хочет: *«Возьми цветовую гамму и общую тему этого ретро-пейзажа и создай эпический кинокадр 4K со свободной детализацией»* (Уровень свободы = 90%).

Поэтому в пайплайне `nn-upscaler` реализован переключатель трёх строгих режимов.

---

## 3. Сравнительная матрица трёх операционных стратегий

```
[ СТРАТЕГИЯ 1: Strict Anchor ]      ──> 100% пиксельное совпадение блоков и сетки
[ СТРАТЕГИЯ 2: Balanced Realism ]    ──> Сглаживание ступеней, точная анатомия и поза (MVP)
[ СТРАТЕГИЯ 3: Free Hallucination ]  ──> Свободный кинематографичный концепт-арт
```

| Параметр диффузии | Стратегия 1: Strict Anchor | Стратегия 2: Balanced Translation | Стратегия 3: Free Hallucination |
|---|---|---|---|
| **Целевое применение** | Модульные тайлы, UI-иконки, блоки | **Игровые персонажи, монстры, экипировка** | Концепт-арт, постеры, фоны |
| **ControlNet-Tile вес ($w_{\text{tile}}$)** | **$0.95 – 1.00$** | **$0.70 – 0.75$** | **$0.30 – 0.40$** |
| **ControlNet-Depth вес ($w_{\text{depth}}$)** | $0.20$ | **$0.35 – 0.40$** | $0.20$ |
| **Denoising Strength ($t_0$)** | $0.55 – 0.65$ | **$0.75 – 0.85$** | $0.90 – 0.95$ |
| **Classifier-Free Guidance (CFG)** | $5.0 – 6.0$ | **$6.5 – 7.5$** | $8.0 – 9.5$ |
| **Шаги отключения (`control_end`)** | $1.00$ (до самого конца) | **$0.85$** (свобода на микропорах) | $0.65$ (раннее отключение) |
| **Предварительное сглаживание** | Минимальное ($\sigma=20$) | **Билатеральное ($\sigma=75$)** | Сильное гауссово ($\sigma=100$) |
| **Сохранение палитры** | 100% жесткое совпадение | Органическое PBR-освещение | Свободная драматическая светотень |

---

## 4. Явление фазового перехода геометрии (Phase Transition)

В диффузионных моделях с ControlNet зависимость геометрической точности от веса адаптера $w_{\text{tile}}$ **нелинейна**:
- При $w_{\text{tile}} < 0.45$: Модель полностью игнорирует форму мелких деталей спрайта (фаза свободной галлюцинации).
- В узком диапазоне **$w_{\text{tile}} \in [0.65, 0.78]$** происходит **фазовый переход**: модель резко фиксирует контур и позу, но сохраняет достаточную генеративную пластичность для превращения ступеней в гладкую кожу и металл.
- При $w_{\text{tile}} > 0.88$: Модель впадает в фазу заморозки, пытаясь воспроизвести даже единичные пиксели-сироты и ступенчатость контуров.

```
       Качество генерации
             ▲
             │             [ ЗОЛОТАЯ ЗОНА: 0.70 - 0.78 ]
Фотореализм  │                   ╭───────────╮
  текстур    │                 ╭─╯           ╰─╮
             │               ╭─╯               ╰─╮
Сетка кубиков│ ──────────────╯                   ╰─────────────
             └─────────────────────────────────────────────────►
             0.0            0.50             0.80           1.0  Вес Tile
             (Хаос/Галлюцинации)             (Заморозка блоков)
```

---

## 5. Практическая реализация: Модуль пресетов в PyTorch

```python
"""nn_upscaler.pipeline.strategy_selector: Preset manager for constraint level strategies."""

from dataclasses import dataclass
from enum import Enum


class ConstraintStrategy(str, Enum):
    STRICT_ANCHOR = "strict_anchor"
    BALANCED_TRANSLATION = "balanced_translation"
    FREE_HALLUCINATION = "free_hallucination"


@dataclass
class StrategyParameters:
    tile_weight: float
    depth_weight: float
    denoise_strength: float
    guidance_scale: float
    controlnet_end_step: float
    bilateral_sigma: int
    description: str


STRATEGY_REGISTRY = {
    ConstraintStrategy.STRICT_ANCHOR: StrategyParameters(
        tile_weight=0.95,
        depth_weight=0.20,
        denoise_strength=0.60,
        guidance_scale=5.5,
        controlnet_end_step=1.0,
        bilateral_sigma=25,
        description="Максимальное удержание каждого блока сетки; для модульных тайлов и UI",
    ),
    ConstraintStrategy.BALANCED_TRANSLATION: StrategyParameters(
        tile_weight=0.75,
        depth_weight=0.35,
        denoise_strength=0.80,
        guidance_scale=7.0,
        controlnet_end_step=0.85,
        bilateral_sigma=75,
        description="Стандарт для персонажей: сглаживание ступеней, точная анатомия и поза",
    ),
    ConstraintStrategy.FREE_HALLUCINATION: StrategyParameters(
        tile_weight=0.35,
        depth_weight=0.20,
        denoise_strength=0.92,
        guidance_scale=8.5,
        controlnet_end_step=0.65,
        bilateral_sigma=100,
        description="Свободная художественная интерпретация по мотивам ретро-концепта",
    ),
}


def get_strategy_parameters(strategy: ConstraintStrategy) -> StrategyParameters:
    return STRATEGY_REGISTRY[strategy]
```

---

## 6. Связанные темы
- Структурные сигналы: [`04_Structure_Preservation/01_Structural_Conditioning_Signals.md`](01_Structural_Conditioning_Signals.md)
- Гибридный пайплайн: [`03_Architectures/07_Hybrid_Multi_Stage_Pipeline.md`](../03_Architectures/07_Hybrid_Multi_Stage_Pipeline.md)
- MVP реализация: [`11_Project_Roadmap/01_MVP_Specification_and_Implementation.md`](../11_Project_Roadmap/01_MVP_Specification_and_Implementation.md)
