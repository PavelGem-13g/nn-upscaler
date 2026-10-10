---
title: "Функции потерь и перцептивные метрики качества"
tags: [math, loss-functions, lpips, dists, dinov2, perceptual-loss, sobel]
---

# Функции потерь и перцептивные метрики качества

## 1. Краткое описание
В этой статье даётся математический анализ функций потерь и перцептивных метрик, используемых для обучения и оценки генеративных систем. Рассматриваются попиксельные потери ($L_1$, $L_2$, Charbonnier), градиентные потери (Sobel, Laplacian), глубокие перцептивные метрики (LPIPS, DISTS) и самоконтролируемые семантические функции потерь на базе трансформера DINOv2. Формулируется композитная функция потерь для задачи Pixel Art → Realism.

---

## 2. Интуитивное объяснение: Слепота попиксельных метрик

Представьте две фотографии кирпичной стены:
- На первом фото стена снята в фокусе.
- На втором фото точно такая же стена снята со сдвигом камеры всего на **1 пиксель вправо**.

Для человеческого глаза обе фотографии абсолютно идентичны — на них видна одна и та же стена с безупречной резкостью.  
Однако классическая среднеквадратичная ошибка ($L_2$ / PSNR) сравнит значения пикселей попарно: тёмный цементный шов наложится на светлый кирпич. В результате PSNR покажет катастрофическое падение качества (например, жалкие 12 dB), словно изображение безнадёжно испорчено шумом!

В задаче **Pixel Art → Realism** это противоречие достигает апогея:
- Модель **обязана** придумать новые поры кожи, волоски и фактуру ткани, которых физически не было в пиксельном квадрате.
- Любая попиксельная метрика жестоко оштрафует модель за появление этих деталей.

Поэтому обучение и оценка генеративных систем требуют метрик, оценивающих **структурную инвариантность и перцептивное подобие**, а не совпадение матриц чисел.

---

## 3. Математический анализ функций потерь

### 3.1. Попиксельные функции потерь (Pixel-level Losses)

#### 1. Средняя абсолютная ошибка ($L_1$ / MAE)
$$\mathcal{L}_{L_1}(x, \hat{x}) = \frac{1}{C H W} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W |x_{c, i, j} - \hat{x}_{c, i, j}|$$
$L_1$-лосс даёт медианную оценку распределения, порождая значительно меньше размытия по сравнению с $L_2$, так как не штрафует квадратично редкие выбросы.

#### 2. Потеря Шарбонье (Charbonnier Loss)
Дифференцируемая сглаженная аппроксимация нормы $L_1$:
$$\mathcal{L}_{\text{Charb}}(x, \hat{x}) = \frac{1}{N} \sum_{k=1}^N \sqrt{(x_k - \hat{x}_k)^2 + \epsilon^2}$$
где $\epsilon = 10^{-3}$ устраняет недифференцируемость производной в нуле.

---

### 3.2. Градиентные и частотные потери (Edge & Gradient Losses)

Чтобы заставить модель сохранять резкие внешние силуэты персонажа без размытия краёв, применяется оператор Собеля (Sobel Operator):
$$G_x = \begin{pmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{pmatrix} * I, \quad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{pmatrix} * I$$
Величина градиента:
$$\nabla I = \sqrt{G_x^2 + G_y^2 + \epsilon}$$

Функция потерь по градиентам границ:
$$\mathcal{L}_{\text{Sobel}}(x, \hat{x}) = \| \nabla x - \nabla \hat{x} \|_1$$

---

### 3.3. Перцептивные метрики (Deep Perceptual Losses)

#### 1. LPIPS (Zhang et al., CVPR 2018)
Метрика LPIPS пропускает изображение через предварительно обученную глубокую сеть (VGG-16 или AlexNet), извлекает пространственные карты активаций на слоях $l \in \{1, \dots, L\}$, нормализует их по L2-норме вдоль каналов и вычисляет взвешенное евклидово расстояние:
$$d_{\text{LPIPS}}(x, \hat{x}) = \sum_{l=1}^L \frac{1}{H_l W_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} \| w_l \odot \left( \hat{\phi}_l(x)_{h, w} - \hat{\phi}_l(\hat{x})_{h, w} \right) \|_2^2$$
где вектор весов $w_l$ откалиброван на масштабном психофизическом датасете BAPPS по суждениям людей.

#### 2. DISTS (Ding et al., IEEE TPAMI 2022) — Золотой стандарт для супер-разрешения
В отличие от LPIPS, метрика DISTS явно разделяет оценку подобия структуры и текстуры через сравнение средних и дисперсий признаков VGG:
$$\mathcal{L}_{\text{DISTS}}(x, \hat{x}) = 1 - \sum_{l=0}^M \left( \alpha_l S(\mu_l(x), \mu_l(\hat{x})) + \beta_l T(\sigma_l(x), \sigma_l(\hat{x})) \right)$$
где $S$ — мера структурного сходства, а $T$ — мера текстурного подобия. DISTS **не штрафует генеративную модель за синтез новых реалистичных микротекстур**, если общая структура сцены сохранена!

---

### 3.4. Семантическая потеря DINOv2 (ViT Patch Cosine Similarity)

Фундаментальная модель DINOv2 (Oquab et al., 2024), обученная без учителя на гигантских корпусах данных, выучивает высокоуровневые семантические представления частей объектов (рука, крыло, колесо, клинок).  
Семантический лосс вычисляется через косинусное сходство пространственных токенов ViT-B/14:
$$\mathcal{L}_{\text{DINO}}(x, \hat{x}) = 1 - \frac{1}{N} \sum_{i=1}^N \frac{\phi_{\text{DINO}}(x)_i \cdot \phi_{\text{DINO}}(\hat{x})_i}{\|\phi_{\text{DINO}}(x)_i\|_2 \|\phi_{\text{DINO}}(\hat{x})_i\|_2}$$

---

## 4. Композитная функция потерь для Pixel Art → Realism

Для эффективного файнтюнинга диффузионных адаптеров и LoRA формулируется сбалансированная функция потерь:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{diff}} \mathcal{L}_{\text{diff}} + \lambda_{\text{dists}} \mathcal{L}_{\text{DISTS}} + \lambda_{\text{dino}} \mathcal{L}_{\text{DINO}} + \lambda_{\text{edge}} \mathcal{L}_{\text{Sobel}}$$

Оптимальные откалиброванные весовые коэффициенты:
- $\lambda_{\text{diff}} = 1.0$ (Базовый диффузионный шум)
- $\lambda_{\text{dists}} = 0.40$ (Перцептивный реализм без текстурного штрафа)
- $\lambda_{\text{dino}} = 0.25$ (Семантическая фиксация классов объектов)
- $\lambda_{\text{edge}} = 0.15$ (Сохранение контуров и позы персонажа)

---

## 5. Практическая реализация: Композитный модуль потерь на PyTorch

```python
"""nn_upscaler.losses.composite_loss: Multi-objective structural and perceptual loss module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class SobelEdgeLoss(nn.Module):
    """Штраф за расхождение внешних силуэтов и контуров."""

    def __init__(self):
        super().__init__()
        # Ядра Собеля
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Переводим в градации серого
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)

        gx_pred = F.conv2d(pred_gray, self.kx, padding=1)
        gy_pred = F.conv2d(pred_gray, self.ky, padding=1)
        edge_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)

        gx_targ = F.conv2d(target_gray, self.kx, padding=1)
        gy_targ = F.conv2d(target_gray, self.ky, padding=1)
        edge_targ = torch.sqrt(gx_targ**2 + gy_targ**2 + 1e-6)

        return F.l1_loss(edge_pred, edge_targ)


class CompositeDepixelizationLoss(nn.Module):
    """Сбалансированная функция потерь для обучения адаптеров Pixel Art -> Realism."""

    def __init__(
        self,
        weight_l1: float = 0.2,
        weight_lpips: float = 0.5,
        weight_edge: float = 0.3,
        device: str = "cuda",
    ):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_lpips = weight_lpips
        self.weight_edge = weight_edge

        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.edge_fn = SobelEdgeLoss().to(device)

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        generated: Сгенерированное изображение [B, 3, H, W] в диапазоне [-1, 1]
        target: Истинное фото высокого разрешения [B, 3, H, W] в диапазоне [-1, 1]
        """
        loss_l1 = F.l1_loss(generated, target)
        loss_lpips = torch.mean(self.lpips_fn(generated, target))
        loss_edge = self.edge_fn(generated, target)

        total_loss = (
            self.weight_l1 * loss_l1 + self.weight_lpips * loss_lpips + self.weight_edge * loss_edge
        )

        return {
            "total_loss": total_loss,
            "loss_l1": loss_l1.detach(),
            "loss_lpips": loss_lpips.detach(),
            "loss_edge": loss_edge.detach(),
        }
```

---

## 6. Связанные темы
- Метрики оценки качества: [`07_Evaluation_and_Benchmarks/01_Metric_Taxonomy_and_Mathematical_Formulation.md`](../07_Evaluation_and_Benchmarks/01_Metric_Taxonomy_and_Mathematical_Formulation.md)
- Обучение LoRA: [`06_Training_and_Fine_Tuning/02_Training_Objectives_and_Loss_Formulation.md`](../06_Training_and_Fine_Tuning/02_Training_Objectives_and_Loss_Formulation.md)
- Библиография (статьи по LPIPS и DISTS): [`REFERENCES.md`](../REFERENCES.md)
