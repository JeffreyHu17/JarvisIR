# Image Degradation Synthesis Toolkit
This toolkit provides four different image degradation effect generation tools, including foggy, night, rainy, and snowy scenes. This README details the environment setup, weight downloads, and usage methods.

## Directory Structure
```
data_synthetic/
├── fog/                  # Foggy degradation model directory
├── night/                # Night degradation model directory
├── rainy/                # Rainy degradation model directory
│   └── GuidedDisent/     # Auxiliary model for rain degradation
│       ├── configs/      # Configuration files
│       └── weights/      # Weight files
├── snow/                 # Snowy degradation model directory
│   └── checkpoints/      # Model checkpoints
├── fog_syn.py            # Foggy degradation synthesis script
├── night_syn.py          # Night degradation synthesis script
├── rainy_syn.py          # Rainy degradation synthesis script
├── snow_syn.py           # Snowy degradation synthesis script
├── run_synthesis.sh      # Automated synthesis process script
└── README.md             # This document
```

## Environment Configuration

### 1. Create and Activate Virtual Environment and Install Dependencies

```bash
cd degradation_synthesis
conda create -n degradation_simulator python=3.8
conda activate degradation_simulator
pip install -r requirements.txt
```

## Download Pre-trained Weights from [Huggingface](https://huggingface.co/LYL1015/JarvisIR).

Before running the scripts, you need to download the corresponding model weight files. Please create the following directory structure and download the weights:
### 1. Foggy Model Weights

The foggy model uses the Depth-Anything-V2-Large model, which will be automatically downloaded on first run. 

### 2. Rainy and Snowy Model Weights
```bash
# Create necessary directories
mkdir -p rainy/GuidedDisent/weights
mkdir -p snow/checkpoints

# Download model weights from huggingface
https://huggingface.co/LYL1015/JarvisIR
```

## Usage Instructions


### Run Individual Degradations


#### Foggy Degradation
```bash
python fog_syn.py --input_dir /path/to/input --output_dir /path/to/output --resize 512 512
```

#### Night Degradation
```bash
python night_syn.py --input_dir /path/to/input --output_dir /path/to/output --resize 512 512
```

#### Rainy Degradation
```bash
python rainy_syn.py --input_dir /path/to/input --output_dir /path/to/output --image_size 512 512
```

#### Snowy Degradation
```bash
python snow_syn.py --input_dir /path/to/input --output_dir /path/to/output --resize 512 512
```

### Batch Generate All Degradations
Use the provided shell script to generate all degradations at once:

```bash
# Add execution permission to the script
chmod +x run_synthesis.sh

# Run with default parameters
bash run_synthesis.sh ./examples ./output

# Or specify parameters: input directory, output directory, image width, image height
bash run_synthesis.sh /path/to/input /path/to/output 512 512

# example
bash run_synthesis.sh ./examples ./output 512 512


```

## Output Directory Structure


After running the script, the following subdirectories will be created in the specified output directory:

```
output/
├── fog/      # Foggy degraded images
├── night/    # Night degraded images
├── wet/      # Wet degraded images
├── rainy/    # Rainy degraded images
└── snow/     # Snowy degraded images
```

## Notes

1. Ensure input images are daytime clear scenes to obtain the best degradation transformation effect
2. Large resolution images may require more GPU memory
3. The foggy model will additionally generate depth maps (if the `--save_depth` option is enabled)
4. The rainy model will generate both wet degradation and raindrop degradation

## Troubleshooting


### Insufficient CUDA Memory

Try reducing the image size, for example: `--resize 256 256`

### Model Weight Loading Failure
Confirm that weight files have been correctly downloaded and placed in the expected directory structure

### Dependency Import Error
Check that all dependencies are correctly installed and ensure compatible versions are used

## Citation


Thanks for the outstanding contributions from the following works. If you use this toolkit in your research, please cite:

```
# night
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Ziteng and Qi, Guo-Jun and Gu, Lin and You, Shaodi and Zhang, Zenghui and Harada, Tatsuya},
    title     = {Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2553-2562}
}

# rainy
@ARTICLE{10070869,
  author={Pizzati, Fabio and Cerri, Pietro and de Charette, Raoul},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Physics-Informed Guided Disentanglement in Generative Networks}, 
  year={2023},
  volume={45},
  number={8},
  pages={10300-10316},
  keywords={Physics;Training;Rendering (computer graphics);Rain;Lenses;Task analysis;Generative adversarial networks;Autonomous driving;adverse weather;adversarial learning;feature disentanglement;GAN;image to image translation;physics-based rendering;robotics;representation learning;vision and rain},
  doi={10.1109/TPAMI.2023.3257486}}

# fog 
@inproceedings{wu2023ridcp,
  title={Ridcp: Revitalizing real image dehazing via high-quality codebook priors},
  author={Wu, Rui-Qi and Duan, Zheng-Peng and Guo, Chun-Le and Chai, Zhi and Li, Chongyi},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={22282--22291},
  year={2023}
}

# snow 
@article{parmar2024one,
  title={One-step image translation with text-to-image models},
  author={Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2403.12036},
  year={2024}
}
```
