# Gradio Demo Guide

This guide provides instructions on how to run the Gradio demo for JarvisIR.

## Environment Setup

Please follow the environment setup instructions in the [SFT Training Guide](./sft_training.md#environment-setup) to create the conda environment and install the necessary dependencies. The same environment can be used for running the Gradio demo.

Make sure you have activated the conda environment:
```bash
conda activate sft_jarvisir
```


## Download Preview Weights

To run the Gradio demo, you need to download the preview weights and place them in the correct location:

1. Download the JarvisIR preview weights from Hugging Face
2. Create the weights directory (if it doesn't exist):
   ```bash
   mkdir -p checkpoints/jarvisir-preview/
   ```
3. Place the downloaded weight files in the `checkpoints/jarvisir-preview/` directory



## Running the Demo

Once the environment is set up and activated, you can run the Gradio demo with the following command from the root directory of the project:

```bash
python gradio_demo.py
```

This will launch a web interface where you can interact with the model.