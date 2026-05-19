#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "einops",
#   "huggingface-hub",
#   "matplotlib",
#   "numpy",
#   "opencv-python-headless",
#   "pillow",
#   "safetensors",
#   "timm",
#   "torch",
#   "torchvision",
#   "transformers",
# ]
# ///

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sims import apply_siglip2_head_mlp, encode_text_templates, openai_imagenet_template


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0)
    return frame.unsqueeze(0)


def noisy_mask_to_box(mask: np.ndarray, min_area: int = 20, pad: int = 2) -> list[int] | None:
    m = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + np.argmax(areas)

    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    H, W = mask.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W - 1, x + w - 1 + pad)
    y2 = min(H - 1, y + h - 1 + pad)

    return [int(x1), int(y1), int(x2), int(y2)]


def shift_box_up(box: list[int] | None, pixels: int) -> list[int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    shift = min(pixels, y1)
    return [x1, y1 - shift, x2, y2 - shift]


def shrink_box(box: list[int] | None, pixels: int) -> list[int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 += pixels
    y1 += pixels
    x2 -= pixels
    y2 -= pixels
    if x1 > x2 or y1 > y2:
        return None
    return [x1, y1, x2, y2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("assets/chase.mp4"))
    parser.add_argument("--text", default="a boat")
    parser.add_argument("--out", type=Path, default=Path("out/sims_vid_boxes.mp4"))
    parser.add_argument("--threshold", type=float, default=0.085)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = total_frames if args.frames is None else min(args.frames, total_frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.out}")

    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version="c-radio_v3-b",
        adaptor_names=["siglip2"],
        progress=True,
        skip_validation=True,
        trust_repo=True,
    )
    model.to(device).eval()

    adaptor = model.adaptors["siglip2"]
    with torch.inference_mode():
        text_features = encode_text_templates(adaptor, args.text, openai_imagenet_template, device)

    try:
        for idx in range(num_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            x = frame_to_tensor(frame_bgr).to(device)
            nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
            x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

            with torch.inference_mode():
                output = model(x, feature_fmt="NCHW")
                _, spatial_features = output["backbone"]
                image_features = apply_siglip2_head_mlp(model, spatial_features)
                patch_features = F.normalize(image_features, dim=1)
                sims = torch.einsum("bchw,nc->bnhw", patch_features, text_features).squeeze()

            mask = (sims.float().cpu().numpy() >= args.threshold).astype(np.uint8)
            low_res_step_y = max(1, round(height / mask.shape[0]))
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            box = noisy_mask_to_box(mask, min_area=args.min_area, pad=args.pad)
            box = shift_box_up(box, low_res_step_y)
            # box = shrink_box(box, 16)

            boxed_frame = frame_bgr.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(boxed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            writer.write(boxed_frame)

            if (idx + 1) % 25 == 0 or idx + 1 == num_frames:
                print(f"processed {idx + 1}/{num_frames} frames")
    finally:
        cap.release()
        writer.release()

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
