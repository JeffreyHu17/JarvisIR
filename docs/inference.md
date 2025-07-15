# Inference Guide

This guide provides instructions for running inference with JarvisIR to analyze degraded images and get restoration task recommendations.

## Prerequisites

### Environment Setup

Follow the [SFT Training Guide](./sft_training.md#environment-setup) to setup environment and activate it:
```bash
conda activate sft_jarvisir
```

### Model Weights

1. Download from [Hugging Face repository](https://huggingface.co/LYL1015/JarvisIR/tree/main/pretrained/preview)
2. Create checkpoint directory: `mkdir -p ./checkpoints/pretrained/preview/`
3. Place downloaded files in the checkpoint directory

## Usage

Basic inference command:

```bash
python inference.py \
    --from_checkpoint ./checkpoints/pretrained/preview/ \
    --image_folder ./data/inference/images/ \
    --save_folder ./data/inference/results/
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--from_checkpoint` | str | **Required** | Path to model checkpoint directory |
| `--image_folder` | str | None | Folder containing images for inference |
| `--save_folder` | str | `data/inference/result` | Folder to save results |
| `--max_generation_length_of_sampling` | int | 384 | Maximum generation length |
| `--seed` | int | 42 | Random seed for reproducibility |

