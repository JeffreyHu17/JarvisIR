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

### 1. Create Model Directory Structure

```bash
# Create directories for models and datasets
cd ..
mkdir -p base_models
```

### 2. Download Base Model Weights

Download the following models to the base_models folder:
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

### 3. Training with xtuner
- Modify the model and data paths in the **llava_8b_full.py** file. 
- Run the following command to perform sft. 
```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train llava_8b_full.py --deepspeed deepspeed_zero2
```

## Acknowledgements
We would like to thank the [XTuner](https://github.com/InternLM/xtuner) team for open-sourcing such an excellent framework. For more fine-tuning methods, please refer to their official documentation and code repository.



