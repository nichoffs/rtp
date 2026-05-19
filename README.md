# RTP Video Segmentation Experiments

This repo contains small scripts for generating text-conditioned RADIO/SigLIP masks, converting those masks to boxes, and using those boxes as prompts for SAM2 or EdgeTAM video propagation.

## Setup

Install dependencies with uv:

```bash
uv sync
```

For SAM2:

```bash
uv sync --extra sam2
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

For EdgeTAM:

```bash
uv sync --extra edgetam
mkdir -p checkpoints configs
curl -L -o checkpoints/edgetam.pt \
  https://huggingface.co/facebook/EdgeTAM/resolve/main/edgetam.pt
curl -L -o configs/edgetam.yaml \
  https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/sam2/configs/edgetam.yaml
```

To install everything:

```bash
uv sync --extra all
```

## Assets

`assets/` is gitignored. Add your input videos locally:

```bash
mkdir -p assets
cp /path/to/video.mp4 assets/chase.mp4
```

Most scripts default to:

```bash
assets/chase.mp4
```

You can pass a different video with `--video`.

## Outputs

`out/` is gitignored. Scripts create it automatically and write videos/JSONL there.

## Common Commands

Generate a thresholded mask preview:

```bash
uv run python sims_vid.py --video assets/chase.mp4 --text "a boat"
```

Generate mask-derived boxes:

```bash
uv run python sims_vid_boxes.py --video assets/chase.mp4 --text "a boat"
```

Run SAM2 tiny with box prompts:

```bash
uv run python sims_vid_sam_points.py \
  --video assets/chase.mp4 \
  --text "a boat" \
  --frames 60
```

Run EdgeTAM with box prompts:

```bash
uv run python sims_vid_edgetam.py \
  --video assets/chase.mp4 \
  --text "a boat" \
  --frames 60
```

## Notes

- The first RADIO run uses `torch.hub` and may download/cache NVlabs/RADIO.
- SAM2 and EdgeTAM video predictors can slow down on long videos because they initialize and carry state over the full frame sequence. Use `--frames 60` for quick comparisons.
- `sims_vid_sam_points.py` and `sims_vid_edgetam.py` currently use the first valid RADIO-derived box as the prompt frame.
