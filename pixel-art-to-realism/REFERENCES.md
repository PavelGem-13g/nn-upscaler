---
title: "Библиография и аннотированный список литературы"
description: "Академические публикации, первоисточники, официальные репозитории и обзор ключевого научного вклада по теме Pixel Art → Realism."
tags: [references, papers, bibliography, academic-research]
---

# Библиография и аннотированный список литературы

В этом документе собраны ключевые академические публикации, препринты, конференции (CVPR, ICCV, ECCV, SIGGRAPH, NeurIPS, ICLR) и проверенные open-source репозитории, лежащие в основе базы знаний.

---

## 1. Пиксель-арт, воксельный рендеринг и специализированные алгоритмы

1. **Depixelizing Pixel Art**
   - *Авторы:* Johannes Kopf, Dani Lischinski.
   - *Публикация:* ACM Transactions on Graphics (SIGGRAPH), 2011.
   - *DOI:* [10.1145/2010324.1964994](https://doi.org/10.1145/2010324.1964994)
   - *Вклад:* Семинальная работа по векторной депикселизации. Построение планарного графа смежности ячеек, эвристическое разрешение диагоналей $2\times 2$, оптимизация сплайнов Безье для гладких контуров.
   - *Связанная статья базы:* [`03_Architectures/01_Classical_and_Regression_SR.md`](03_Architectures/01_Classical_and_Regression_SR.md)

2. **Voxify3D: Pixel Art Meets Volumetric Rendering**
   - *Авторы:* Yi-Chuan Huang, Jiewen Chan, Hao-Jen Chien, Yu-Lun Liu.
   - *Публикация:* CVPR 2026 (arXiv:2512.07834), 2025/2026.
   - *Ссылка:* [arXiv:2512.07834](https://arxiv.org/abs/2512.07834) | [Project Page](https://yichuanh.github.io/Voxify-3D/)
   - *Вклад:* Двухстадийный фреймворк ортогонального пиксельного контроля, дифференцируемое квантование палитры через Gumbel-Softmax и patch-based CLIP alignment для связки 2D-спрайтов с 3D-волуметрическим представлением.

3. **Texel Splatting: Perspective-Stable 3D Pixel Art**
   - *Автор:* Dylan Ebert.
   - *Публикация:* arXiv:2603.14587, 2026.
   - *Ссылка:* [arXiv:2603.14587](https://arxiv.org/abs/2603.14587)
   - *Вклад:* Перспективно-стабильный рендеринг пиксель-арта через кубические карты и сплайтинг текселей в мировом пространстве без дрейфа субпикселей при движении камеры.

4. **Pixel VQ-VAEs for Improved Pixel Art Representation**
   - *Авторы:* Akash Saravanan, Matthew Guzdial.
   - *Публикация:* Experimental AI in Games Workshop (EXAG), 2022 (arXiv:2203.12130).
   - *Ссылка:* [arXiv:2203.12130](https://arxiv.org/abs/2203.12130)
   - *Вклад:* Исследование представлений пиксель-арта через дискретный VQ-VAE кодовый словарь; доказательство неприменимости стандартных CNN без адаптации под пиксели-блоки.

5. **Analysis and Compilation of Normal Map Generation Techniques for Pixel Art**
   - *Авторы:* Rodrigo D. Moreira, Flávio Coutinho, Luiz Chaimowicz.
   - *Публикация:* IEEE SBGames, 2022 (arXiv:2212.09692).
   - *DOI:* [10.1109/SBGAMES56371.2022.9961116](https://doi.org/10.1109/SBGAMES56371.2022.9961116)
   - *Вклад:* Сравнительный анализ методов синтеза карт нормалей для 2D-спрайтов с целью физически корректного динамического освещения в играх.

---

## 2. Диффузионные модели и Flow Matching

6. **Denoising Diffusion Probabilistic Models (DDPM)**
   - *Авторы:* Jonathan Ho, Ajay Jain, Pieter Abbeel.
   - *Публикация:* NeurIPS 2020.
   - *Ссылка:* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
   - *Вклад:* Фундамент современных диффузионных моделей. Параметризация предсказания шума $\epsilon_\theta$, связь с вариационными нижними оценками (ELBO).

7. **High-Resolution Image Synthesis with Latent Diffusion Models (LDM / Stable Diffusion)**
   - *Авторы:* Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.
   - *Публикация:* CVPR 2022.
   - *Ссылка:* [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion)
   - *Вклад:* Перенос диффузии в латентное пространство VAE, введение механизмов перекрёстного внимания (Cross-Attention) для мультимодального кондиционирования.

8. **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**
   - *Авторы:* Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann et al.
   - *Публикация:* ICLR 2024.
   - *Ссылка:* [arXiv:2307.01952](https://arxiv.org/abs/2307.01952) | [Stability AI](https://github.com/Stability-AI/generative-models)
   - *Вклад:* Трёхкратное увеличение бэкбона U-Net, двойной текстовый энкодер (OpenCLIP ViT-G/14 + CLIP ViT-L/14), микро-кондиционирование по размеру и координатам обрезки.

9. **Flow Straight and Fast: Learning to Generate and Transfer Data with Flow Matching**
   - *Авторы:* Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le.
   - *Публикация:* ICLR 2023.
   - *Ссылка:* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
   - *Вклад:* Математическая теория Flow Matching на основе непрерывных динамических систем (ODE). Основа для моделей FLUX.1 и Stable Diffusion 3.

---

## 3. Структурный контроль, адаптеры и инверсия

10. **Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)**
    - *Авторы:* Lvmin Zhang, Anyi Rao, Maneesh Agrawala.
    - *Публикация:* ICCV 2023.
    - *Ссылка:* [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | [GitHub](https://github.com/lllyasviel/ControlNet)
    - *Вклад:* Архитектура замороженного U-Net с параллельной обучаемой веткой и zero-convolutions. Революция в пространственном управлении генерацией (Canny, Depth, Tile, Lineart).

11. **IP-Adapter: Text-Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**
    - *Авторы:* Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, Wei Yang.
    - *Публикация:* arXiv:2308.06721, 2023.
    - *Ссылка:* [arXiv:2308.06721](https://arxiv.org/abs/2308.06721) | [GitHub](https://github.com/tencent-ailab/IP-Adapter)
    - *Вклад:* Развязанный механизм перекрёстного внимания (Decoupled Cross-Attention), позволяющий независимо подавать текстовый промпт и визуальный референс без деградации структуры.

12. **MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Generation**
    - *Авторы:* Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, Yinqiang Zheng.
    - *Публикация:* ICCV 2023.
    - *Ссылка:* [arXiv:2304.08465](https://arxiv.org/abs/2304.08465) | [GitHub](https://github.com/TencentARC/MasaCtrl)
    - *Вклад:* Управление самовниманием без дообучения сети; перенос ключей ($K$) и значений ($V$) между исходным и сгенерированным кадром для строгой фиксации геометрии.

13. **FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model**
    - *Авторы:* Sicheng Mo, Fangzhou Mu, Kuan Wei Huang, Ji-Liang Le, Peng Wang.
    - *Публикация:* CVPR 2024.
    - *Ссылка:* [arXiv:2312.07536](https://arxiv.org/abs/2312.07536)
    - *Вклад:* Анализ спектра PCA в слоях самовнимания; пространственное управление генерацией без необходимости обучения ControlNet-адаптеров.

---

## 4. Диффузионная реставрация и супер-разрешение

14. **DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior**
    - *Авторы:* Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, Chao Dong.
    - *Публикация:* arXiv:2308.15070, 2023/2024.
    - *Ссылка:* [arXiv:2308.15070](https://arxiv.org/abs/2308.15070) | [GitHub](https://github.com/XPixelGroup/DiffBIR)
    - *Вклад:* Двухстадийный пайплайн: 1) Регрессионное удаление деградации; 2) Генерация реалистичных деталей через IRControlNet с настраиваемым restoration-guidance балансом.

15. **SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution**
    - *Авторы:* Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, Lei Zhang.
    - *Публикация:* CVPR 2024.
    - *Ссылка:* [arXiv:2311.16518](https://arxiv.org/abs/2311.16518) | [GitHub](https://github.com/cswry/SeeSR)
    - *Вклад:* Автоматическое извлечение устойчивых семантических промптов из сильно деградированных входов и подмешивание низкоразмерного изображения в начальный шум для подавления галлюцинаций.

16. **Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild (SUPIR)**
    - *Авторы:* Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, Chao Dong.
    - *Публикация:* CVPR 2024.
    - *Ссылка:* [arXiv:2401.13627](https://arxiv.org/abs/2401.13627) | [GitHub](https://github.com/Fanghua-Yu/SUPIR)
    - *Вклад:* Масштабирование реставрации до SDXL, обучение на 20 миллионах изображений, мультимодальное управление текстом и restoration-guided sampling.

17. **Towards Robust Blind Face Restoration with Codebook Lookup Transformer (CodeFormer)**
    - *Авторы:* Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, Chen Change Loy.
    - *Публикация:* NeurIPS 2022.
    - *Ссылка:* [arXiv:2206.11253](https://arxiv.org/abs/2206.11253) | [GitHub](https://github.com/sczhou/CodeFormer)
    - *Вклад:* Использование дискретного кодового словаря VQ-GAN для восстановления поврежденных и пикселизированных лиц с регулируемым балансом fidelity vs quality.

---

## 5. Перцептивные метрики качества

18. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)**
    - *Авторы:* Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
    - *Публикация:* CVPR 2018.
    - *Ссылка:* [arXiv:1801.03924](https://arxiv.org/abs/1801.03924) | [GitHub](https://github.com/richzhang/PerceptualSimilarity)
    - *Вклад:* Фундаментальная перцептивная метрика, основанная на расстояниях признаков глубоких сетей (AlexNet, VGG), отражающая человеческое зрительное восприятие.

19. **Image Quality Assessment Based on Deep Image Structure and Texture Similarity (DISTS)**
    - *Авторы:* Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli.
    - *Публикация:* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022.
    - *Ссылка:* [arXiv:2004.07728](https://arxiv.org/abs/2004.07728) | [GitHub](https://github.com/dingkeyan93/DISTS)
    - *Вклад:* Явное разделение перцептивной метрики на инвариантность к текстурным трансформациям и чувствительность к структурным искажениям.

20. **DINOv2: Learning Robust Visual Features without Supervision**
    - *Авторы:* Maxime Oquab, Timothée Darcet, Théo Moutakanni et al.
    - *Публикация:* Transactions on Machine Learning Research (TMLR), 2024 (arXiv:2304.07193).
    - *Ссылка:* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | [GitHub](https://github.com/facebookresearch/dinov2)
    - *Вклад:* Самоконтролируемые визуальные признаки Foundation модели ViT, обеспечивающие надёжное сопоставление плотных семантических частей объектов между спрайтом и реалистичным фото.

---

## 6. Видео и Real-Time генерация

21. **CoDeF: Content Deformation Fields for Temporally Consistent Video Processing**
    - *Авторы:* Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, Yujun Shen.
    - *Публикация:* CVPR 2024.
    - *Ссылка:* [arXiv:2308.07926](https://arxiv.org/abs/2308.07926) | [GitHub](https://github.com/qiuyu96/CoDeF)
    - *Вклад:* Представление видео в виде единого канонического изображения и поля деформаций во времени, обеспечивающее полную ликвидацию фликкеринга при видео-трансляции.

22. **StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation**
    - *Авторы:* Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Gomi et al.
    - *Публикация:* arXiv:2312.12491, 2023/2024.
    - *Ссылка:* [arXiv:2312.12491](https://arxiv.org/abs/2312.12491) | [GitHub](https://github.com/cumulo-autumn/StreamDiffusion)
    - *Вклад:* Конвейеризация процесса сэмплирования диффузии (Stream Batch), слияние VAE-инференса и использование TensorRT, позволяющее достигать $>50$ FPS при инференсе на современных GPU.
