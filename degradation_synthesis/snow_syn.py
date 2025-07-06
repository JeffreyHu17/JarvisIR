"""
Usage Examples:

1. Basic usage:
python snow_syn.py --input_dir /path/to/clear/images --output_dir /path/to/output

2. Using default parameters:
python snow_syn.py
This will use default input directory '/home/wkr/workspace/workspace_lyl/datasets/mini_day_night_data/eval/day'
and default output directory './output'

3. Resize images:
python snow_syn.py --resize 256 256

The program will convert clear images to snowy images and save them in output_dir/snow/ directory.
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from snow.cyclegan_turbo import CycleGAN_Turbo
import argparse
from tqdm import tqdm

def load_image(image_path, resize=None):
    """Load image and convert to tensor format"""
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.5], [0.5]))
    
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def process_directory(input_dir, output_dir, model_cfg, resize=None):
    """Process all images in a directory"""
    # Check if CUDA device is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create snow processor
    model = CycleGAN_Turbo(
        pretrained_name=model_cfg['model_name'],
        pretrained_path=model_cfg['model_path']
    ).to(device)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, 'snow'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create progress bar using tqdm
    for img_file in tqdm(image_files, desc="Processing images", ncols=100):
        # Load image
        input_path = os.path.join(input_dir, img_file)
        img_tensor = load_image(input_path, resize=resize).to(device)
        
        # Generate snowy image
        with torch.no_grad():
            snow_img = model(img_tensor, direction=model_cfg['direction'], caption=model_cfg['prompt'])
        
        # Move back to CPU for saving
        snow_img = snow_img.cpu()
        
        # Save results
        output_name = os.path.splitext(img_file)[0]
        snow_pil = transforms.ToPILImage()(snow_img[0] * 0.5 + 0.5)
        snow_pil.save(os.path.join(output_dir, 'snow', f'{output_name}.png'))

def main():
    parser = argparse.ArgumentParser(description='Snow image synthesis test program')
    parser.add_argument('--input_dir', default='./examples',
                        type=str, help='Input image directory')
    parser.add_argument('--output_dir', default='./output', type=str, help='Output image directory')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512], help='Resize images to specified resolution, default is 512x512')
    parser.add_argument('--model_name', type=str, default="clear_to_snow", help='name of the pretrained model')
    parser.add_argument('--model_path', type=str, default="./snow/checkpoints/day2snow.pkl", help='path to model weights')
    parser.add_argument('--prompt', type=str, default='driving in heavy snow', help='prompt for generation')
    parser.add_argument('--direction', type=str, default='a2b', help='translation direction')
    
    args = parser.parse_args()
    
    # Configure model parameters
    model_cfg = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'prompt': args.prompt,
        'direction': args.direction
    }
    
    # Process directory
    resize = tuple(args.resize) if args.resize else None
    process_directory(args.input_dir, args.output_dir, model_cfg, resize=resize)

if __name__ == '__main__':
    main()
