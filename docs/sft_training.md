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

cd ../../src/sft/xtuner

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

- Modify the contents of lines 31 to 48 of the configuration file(**src/sft/llava_8b_full.py**):  
``` python
cmd_args = lambda: None
cmd_args.llm_name_or_path = 'llava-llama-3-8b' # Path to the LLM model
cmd_args.visual_encoder_name_or_path = 'clip-vit-large-patch14-336' # Path to the visual encoder model
cmd_args.pretrained_pth = 'llava-llama-3-8b-pretrain/iter_2181.pth' # Path to the pretrained model weights
cmd_args.data_path = None # Path to the training data JSON file
cmd_args.image_folder = None # Path to the image folder
cmd_args.evaluation_images = None # Path to the evaluation images
cmd_args.max_epochs = None # Maximum number of training epochs
cmd_args.batch_size = None # Batch size per device
cmd_args.accumulative_counts = None # Gradient accumulation steps
cmd_args.dataloader_num_workers = None # Number of dataloader workers
cmd_args.lr = None # Learning rate
cmd_args.weight_decay = None # Weight decay
cmd_args.max_norm = None # Gradient clipping max norm
cmd_args.warmup_ratio = None # Warmup ratio
cmd_args.save_steps = None # Save checkpoint every X steps
cmd_args.save_total_limit = None # Maximum number of checkpoints to keep
cmd_args.evaluation_freq = None # Evaluation frequency in iterations
```
- Use the following command:  
```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train Absolute_Path/src/sft/llava_8b_full.py --deepspeed deepspeed_zero2 
```

<!-- ### 3. Training with our pretrained weights

- We provide pretrained JarvisIR model weights on [HuggingFace](https://huggingface.co/LYL1015/JarvisIR/tree/main/pretrained/preview). Put them in your local directory structure (e.g., `./checkpoints/pretrained_models/preview`).
- Modify the contents of lines 31 to 48 of the configuration file(**src/sft/llava_8b_full.py**)
``` python
cmd_args = lambda: None
cmd_args.llm_name_or_path = None # Path to the LLM model
cmd_args.visual_encoder_name_or_path = None # Path to the visual encoder model
cmd_args.pretrained_pth = './checkpoints/pretrained_models/preview' # Path to the pretrained model weights
cmd_args.data_path = None # Path to the training data JSON file
cmd_args.image_folder = None # Path to the image folder
cmd_args.evaluation_images = None # Path to the evaluation images
cmd_args.max_epochs = None # Maximum number of training epochs
cmd_args.batch_size = None # Batch size per device
cmd_args.accumulative_counts = None # Gradient accumulation steps
cmd_args.dataloader_num_workers = None # Number of dataloader workers
cmd_args.lr = None # Learning rate
cmd_args.weight_decay = None # Weight decay
cmd_args.max_norm = None # Gradient clipping max norm
cmd_args.warmup_ratio = None # Warmup ratio
cmd_args.save_steps = None # Save checkpoint every X steps
cmd_args.save_total_limit = None # Maximum number of checkpoints to keep
cmd_args.evaluation_freq = None # Evaluation frequency in iterations
```
- Use the following command: 
```bash
NPROC_PER_NODE=${GPU_NUM} xtuner train Absolute_Path/src/sft/llava_8b_full.py --deepspeed deepspeed_zero2 
``` -->


## Acknowledgements
We would like to thank the [XTuner](https://github.com/InternLM/xtuner) team for open-sourcing such an excellent framework. For more fine-tuning methods, please refer to their official documentation and code repository.



