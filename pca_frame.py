#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "einops",
#   "huggingface-hub",
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
from PIL import Image


def read_frame(video_path: Path, frame_index: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0)
    return frame.unsqueeze(0)


def pca_rgb(features: torch.Tensor) -> Image.Image:
    if features.ndim != 4 or features.shape[0] != 1:
        raise ValueError(f"Expected features shaped (1, C, H, W), got {tuple(features.shape)}")

    _, _, h, w = features.shape
    x = features[0].permute(1, 2, 0).reshape(h * w, -1).float().cpu()
    x = x - x.mean(dim=0, keepdim=True)

    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    rgb = (x @ v[:, :3]).reshape(h, w, 3)

    lo = torch.quantile(rgb.reshape(-1, 3), 0.01, dim=0)
    hi = torch.quantile(rgb.reshape(-1, 3), 0.99, dim=0)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    return Image.fromarray((rgb.numpy() * 255).astype(np.uint8))


def apply_siglip2_head_mlp(model, features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    adaptor = model.adaptors["siglip2"]
    first_param = next(adaptor.head_mlp.parameters())

    x = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
    x = adaptor.head_mlp(x.to(dtype=first_param.dtype)).to(dtype=features.dtype)
    return x.reshape(b, h, w, x.shape[-1]).permute(0, 3, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("assets/chase.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/pca_frame.png"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--language-alignment", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version="c-radio_v3-b",
        adaptor_names=["siglip2"] if args.language_alignment else None,
        progress=True,
        skip_validation=True,
        trust_repo=True,
    )
    model.to(device).eval()

    x = read_frame(args.video, args.frame).to(device)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

    with torch.inference_mode():
        output = model(x, feature_fmt="NCHW")
        _, spatial_features = output["backbone"] if args.language_alignment else output
        if args.language_alignment:
            spatial_features = apply_siglip2_head_mlp(model, spatial_features)

    print(f"spatial_features shape: {tuple(spatial_features.shape)}")

    img = pca_rgb(spatial_features)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
