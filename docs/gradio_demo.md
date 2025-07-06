# Gradio Demo Guide

This guide provides instructions on how to run the Gradio demo for JarvisIR.

## Environment Setup

Please follow the environment setup instructions in the [SFT Training Guide](./sft_training.md#environment-setup) to create the conda environment and install the necessary dependencies. The same environment can be used for running the Gradio demo.

Make sure you have activated the conda environment:
```bash
conda activate sft_jarvisir
```


## Download Preview Weights

To run the Gradio demo, you need to download the preview weights from Hugging Face and place them in the correct location:

1. Download the JarvisIR preview weights from [Hugging Face repository](https://huggingface.co/LYL1015/JarvisIR/tree/main/pretrained/preview)
2. Create the weights directory (if it doesn't exist):
   ```bash
   cd JarvisIR/
   mkdir -p ./checkpoints/pretrained/preview/
   ```
3. Place the downloaded weight files in the `./checkpoints/pretrained/preview/` directory


## Running the Demo

Once the environment is set up and activated, you can run the Gradio demo with the following command from the root directory of the project:

```bash
python demo_gradio.py --model_path ./checkpoints/pretrained/preview/ --cuda_device 0
```

The command accepts the following parameters:
- `--model_path`: Required parameter specifying the path to the LLaVA model (mandatory)
- `--cuda_device`: Optional parameter to specify which CUDA device to use (default is 0)

This will launch a web interface at `http://0.0.0.0:7866` where you can interact with the model.

## Important Notes

- Make sure the model path points to a directory containing a valid LLaVA model with all necessary files
- The web interface will be accessible on port 7866
- The tool engine requires specific restoration models which should be installed as part of the environment setup
