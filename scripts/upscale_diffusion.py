"""scripts/upscale_diffusion.py: Production CLI tool for Pixel Art to Realism conversion."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image

from nn_upscaler.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Pixel Art to Realism Upscaler")
    parser.add_argument(
        "--input", type=str, required=True, help="Путь к входному пиксель-арт изображению"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/result_realistic.png", help="Путь для сохранения"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="photorealistic highly detailed photograph, 8k resolution, raw photo, physical materials",
        help="Текстовое описание сцены",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="pixelated, low resolution, 8-bit, blocky, cartoon, blurry, bad anatomy",
        help="Негативный промпт",
    )
    parser.add_argument(
        "--tile-weight", type=float, default=0.75, help="Вес адаптера ControlNet-Tile (0.5 - 1.0)"
    )
    parser.add_argument("--steps", type=int, default=28, help="Число шагов сэмплирования")
    parser.add_argument("--seed", type=int, default=42, help="Случайный сид")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    dtype = torch.float32 if device.type == "mps" else torch.float16

    print(f"[*] Старт пайплайна Pixel Art -> Realism на {device} ({dtype})...")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл {input_path} не найден!")

    raw_sprite = Image.open(input_path).convert("RGB")

    # 1. Этап предобработки: масштабирование и билатеральный фильтр
    target_size = (1024, 1024)
    upscaled_nearest = raw_sprite.resize(target_size, Image.Resampling.NEAREST)
    arr_nearest = np.array(upscaled_nearest)
    smoothed = cv2.bilateralFilter(arr_nearest, d=9, sigmaColor=75, sigmaSpace=75)
    condition_image = Image.fromarray(smoothed)

    # 2. Инициализация моделей
    print("[*] Загрузка весов SDXL и ControlNet-Tile...")
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-tile-sdxl-1.0",
        torch_dtype=dtype,
    )

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
    ).to(device)

    # 3. Настройка планировщика и оптимизации памяти
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )

    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    elif device.type == "mps":
        pipe.enable_vae_tiling()

    # 4. Генеративный инференс
    print(f"[*] Генерация фотореализма ({args.steps} шагов, вес Tile={args.tile_weight})...")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=condition_image,
        controlnet_conditioning_scale=args.tile_weight,
        num_inference_steps=args.steps,
        guidance_scale=7.0,
        generator=generator,
    ).images[0]

    # 5. Сохранение результата
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"[+] Фотореалистичное изображение успешно сохранено в: {out_path}")


if __name__ == "__main__":
    main()
