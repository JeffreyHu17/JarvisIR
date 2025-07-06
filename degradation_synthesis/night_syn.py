"""
Usage Examples:

1. Basic usage:
python night_syn.py --input_dir /path/to/day/images --output_dir /path/to/output

2. Using default parameters:
python night_syn.py
This will use default input directory '/home/wkr/workspace/workspace_lyl/datasets/mini_day_night_data/eval/day'
and default output directory './output'

3. Resize images:
python night_syn.py --resize 256 256

The program will convert daytime images to nighttime images and save them in output_dir/night/ directory
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from night.night_simulator import ImageProcessor, save_image
import argparse
from tqdm import tqdm

def load_image(image_path, resize=None):
    """Load image and convert to tensor format"""
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def process_directory(input_dir, output_dir, degradation_cfg, resize=None):
    """Process all images in a directory"""
    # Check if CUDA device is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create image processor
    processor = ImageProcessor(degradation_cfg)
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, 'night'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create progress bar using tqdm
    for img_file in tqdm(image_files, desc="Processing images", ncols=100):
        # Load image
        input_path = os.path.join(input_dir, img_file)
        img_tensor = load_image(input_path, resize=resize).to(device)
        
        # Generate night image
        night_img = processor.Low_Illumination_Degrading(img_tensor)
        
        # Move back to CPU for saving
        night_img = night_img.cpu()
        
        # Save results
        output_name = os.path.splitext(img_file)[0]
        save_image(night_img[0], os.path.join(output_dir, 'night', f'{output_name}.png'))

def main():
    parser = argparse.ArgumentParser(description='Image degradation test program')
    parser.add_argument('--input_dir', default='./examples',
                        type=str, help='Input image directory')
    parser.add_argument('--output_dir', default='./output', type=str, help='Output image directory')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help='Resize images to specified resolution, default is 512x512')
    
    args = parser.parse_args()
    
    # Configure degradation parameters
    degradation_cfg = {
        'gamma_range': [2.0, 3.5],
        'rgb_range': [0.8, 0.1],
        'red_range': [1.9, 2.4],
        'blue_range': [1.5, 1.9],
        'darkness_range': [0.07, 0.15],
        'quantisation': [2, 4, 6]
    }
    
    # Process directory
    resize = tuple(args.resize) if args.resize else None
    process_directory(args.input_dir, args.output_dir, degradation_cfg, resize=resize)

if __name__ == '__main__':
    main() 