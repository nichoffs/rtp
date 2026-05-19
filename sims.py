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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]


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


def apply_siglip2_head_mlp(model, features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    adaptor = model.adaptors["siglip2"]
    first_param = next(adaptor.head_mlp.parameters())

    x = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
    x = adaptor.head_mlp(x.to(dtype=first_param.dtype)).to(dtype=features.dtype)
    return x.reshape(b, h, w, x.shape[-1]).permute(0, 3, 1, 2)


def encode_text_templates(adaptor, class_name: str, templates, device: str) -> torch.Tensor:
    prompts = [template(class_name) for template in templates]
    text_input = adaptor.tokenizer(prompts).to(device)
    text_features = adaptor.encode_text(text_input, normalize=True)
    text_features = text_features.mean(dim=0, keepdim=True)
    return F.normalize(text_features, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("assets/chase.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--text", default="a car")
    parser.add_argument("--out", type=Path, default=Path("out/sims.png"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
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
        adaptor_names=["siglip2"],
        progress=True,
        skip_validation=True,
        trust_repo=True,
    )
    model.to(device).eval()

    x = read_frame(args.video, args.frame).to(device)
    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

    adaptor = model.adaptors["siglip2"]

    with torch.inference_mode():
        output = model(x, feature_fmt="NCHW")
        _, spatial_features = output["backbone"]
        image_features = apply_siglip2_head_mlp(model, spatial_features)
        text_features = encode_text_templates(adaptor, args.text, openai_imagenet_template, device)

        patch_features = F.normalize(image_features, dim=1)
        sims = torch.einsum("bchw,nc->bnhw", patch_features, text_features).squeeze()

    sim = sims.float().cpu()
    print(f"image_features shape: {tuple(image_features.shape)}")
    print(f"text_features shape: {tuple(text_features.shape)}")
    print(f"cosine similarity range: {sim.min().item():.4f} to {sim.max().item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if args.threshold is None:
        im = ax.imshow(sim.numpy(), cmap="magma")
        fig.colorbar(im, ax=ax, label="cosine similarity")
    else:
        mask = sim.numpy() >= args.threshold
        im = ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="selected")
        print(f"threshold: {args.threshold:.4f}")
        print(f"selected patches: {int(mask.sum())}")
    ax.set_title(args.text)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
