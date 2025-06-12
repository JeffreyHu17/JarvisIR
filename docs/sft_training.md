# SFT (Supervised Fine-Tuning) Training Guide

This guide provides step-by-step instructions for setting up the environment and performing supervised fine-tuning using XTuner framework.

## Environment Setup

### 1. Create and Activate Virtual Environment

```bash
conda create -n sft_jarvisir python=3.10
conda activate sft_jarvisir
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
cd package/agent_tools/Retinexformer &&  python3 setup.py develop --no_cuda_ext && cd ..
cd RIDCP &&  python3 setup.py develop && cd ../..
pip install -e .

cd ../dependences/BasicSR
pip install -e .

cd ../src/sft/xtuner

# Install required packages
pip install -r requirements.txt

# Install XTuner with all features
pip install -e '.[all]'
```

### 3. Verify Installation

```bash
# Check XTuner installation
xtuner version

# Verify GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

## Model Weights Setup

<!-- ### 1. Create Model Directory Structure -->

<!-- ```bash
# Create directories for models and datasets
cd ..
mkdir -p ./checkpoints/base_models
``` -->

### 1. Download Base Model Weights

Download the following models to the ``./checkpoints/base_models`` folder:
<table>
    <tr>
        <th>Model</th><th>Model Weights</th>
    </tr>
    <tr style="border-top: 2px solid">
        <td>openai/clip-vit-large-patch14-336</td><td><a href="https://huggingface.co/openai/clip-vit-large-patch14-336"> 🤗 HuggingFace</a></td>
    </tr>
    <tr style="border-top: 2px solid">
        <td>xtuner/llava-llama-3-8b</td><td><a href="https://huggingface.co/xtuner/llava-llama-3-8b"> 🤗 HuggingFace</a></td>
    </tr>
    <tr style="border-top: 2px solid">
        <td>xtuner/llava-llama-3-8b-pretrain</td><td><a href="https://huggingface.co/xtuner/llava-llama-3-8b-pretrain"> 🤗 HuggingFace</a></td>
    </tr>
</table>

### 2. Training from scratch with base checkpoints

XTuner allows you to specify model paths directly from the command line without modifying the configuration file. Use the following command pattern:

```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train src/sft/llava_8b_full.py \
    --deepspeed deepspeed_zero2 \
    --llm-name-or-path ./checkpoints/base_models/llava-llama-3-8b \
    --visual-encoder-name-or-path ./checkpoints/base_models/clip-vit-large-patch14-336 \
    --data-path data/CleanBench/train/sft_data.json \
    --image-folder data/CleanBench/train/images \
    --evaluation-images data/CleanBench/eval/images/night/000001.png
```

### 3. Training with our pretrained weights

We provide pretrained JarvisIR model weights on [HuggingFace](https://huggingface.co/LYL1015/JarvisIR/tree/main/pretrained/preview). Put them in your local directory structure (e.g., `./checkpoints/pretrained_models/preview`).

```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train src/sft/llava_8b_full.py \
    --deepspeed deepspeed_zero2 \
    --llm-name-or-path ./checkpoints/base_models/llava-llama-3-8b \
    --visual-encoder-name-or-path ./checkpoints/base_models/clip-vit-large-patch14-336 \
    --pretrained-pth ./checkpoints/pretrained_models/preview \
    --data-path data/CleanBench/train/sft_data.json \
    --image-folder data/CleanBench/train/images \
    --evaluation-images data/CleanBench/eval/images/night/000001.png
```


## Acknowledgements
We would like to thank the [XTuner](https://github.com/InternLM/xtuner) team for open-sourcing such an excellent framework. For more fine-tuning methods, please refer to their official documentation and code repository.



