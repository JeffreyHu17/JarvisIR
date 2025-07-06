"""
Usage Examples:

1. Basic usage:
python fog_syn.py --input_dir /path/to/day/images --output_dir /path/to/output

2. Using default parameters:
python fog_syn.py
This will use default input directory '/home/wkr/workspace/workspace_lyl/datasets/mini_day_night_data/eval/day'
and default output directory './output'

3. Save depth maps:
python fog_syn.py --save_depth

4. Resize images:
python fog_syn.py --resize 256 256

The program will convert clear images to foggy images and save them in output_dir/fog/ directory.
If --save_depth is enabled, depth maps will also be saved in the same directory.
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from fog.fog_simulator import ImageProcessor, save_image
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
    os.makedirs(os.path.join(output_dir, 'fog'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create progress bar with tqdm
    for img_file in tqdm(image_files, desc="Processing images", ncols=100):
        # Load image
        input_path = os.path.join(input_dir, img_file)
        img_tensor = load_image(input_path, resize=resize).to(device)
        
        # Generate foggy image
        fog_img = processor.Fog_Degrading(img_tensor)
        
        # Move back to CPU for saving
        fog_img = fog_img.cpu()
        
        # Save results
        output_name = os.path.splitext(img_file)[0]
        save_image(fog_img[0], os.path.join(output_dir, 'fog', f'{output_name}.png'))
        
        # Optional: save depth map visualization
        if degradation_cfg.get('save_depth', False):
            # Convert tensor to numpy
            img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype('uint8')
            # RGB -> BGR
            img_bgr = img_np[:, :, ::-1].copy()
            
            # Generate depth map
            depth = processor.generate_depth(img_bgr)
            # Visualize depth map
            colored_depth = processor.visualize_depth(depth)
            # Convert back to PIL image and save
            colored_depth = (colored_depth * 255).astype('uint8')
            # 创建depth文件夹
            os.makedirs(os.path.join(output_dir, 'fog', 'depth'), exist_ok=True)
            Image.fromarray(colored_depth).save(
                os.path.join(output_dir, 'fog', 'depth', f'{output_name}.png'))

def main():
    parser = argparse.ArgumentParser(description='Fog image synthesis test program')
    parser.add_argument('--input_dir', default='./examples',
                        type=str, help='Input image directory')
    parser.add_argument('--output_dir', default='./output', type=str, help='Output image directory')
    parser.add_argument('--save_depth', type=bool, default=True, help='Whether to save depth maps')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help='Resize images to specified resolution, default is 512x512')
    
    args = parser.parse_args()
    
    # Configure fog generation parameters
    degradation_cfg = {
        'depth_model': 'depth-anything/Depth-Anything-V2-Large-hf',  # Depth model to use
        'save_depth': args.save_depth,  # Whether to save depth maps
        'device': "cuda" if torch.cuda.is_available() else "cpu",  # Add device configuration
        'beta_range': [0.3, 1.5],  # Transmission coefficient range
        'A_range': [0.25, 1.0],  # Atmospheric light range
        'color_p': 1.0,  # Color shift probability
        'color_range': [-0.025, 0.025]  # Color shift range
    }
    # Process directory
    resize = tuple(args.resize) if args.resize else None
    process_directory(args.input_dir, args.output_dir, degradation_cfg, resize=resize)

if __name__ == '__main__':
    main() 