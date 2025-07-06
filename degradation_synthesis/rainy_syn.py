"""
Usage Examples:

1. Basic usage:
python rainy_syn.py --input_dir /path/to/input/images --output_dir /path/to/output

2. Using default parameters:
python rainy_syn.py
This will use default input directory '/home/wkr/workspace/workspace_lyl/datasets/mini_day_night_data/eval/day'
and default output directory './output'

3. Resize images:
python rainy_syn.py --image_size 256 256

The program will generate wet and rainy effects for input images and save them in output_dir/wet/ and output_dir/rainy/ directories
"""

import torch
import yaml
from rainy.GuidedDisent.MUNIT.model_infer import MUNIT_infer
from rainy.GuidedDisent.droprenderer import DropModel
from torchvision import transforms
from PIL import Image
import os
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

class RainDropProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._init_model()
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
    def _init_model(self):
        """Initialize MUNIT model"""
        with open(self.config['params_path'], 'r') as stream:
            hyperparameters = yaml.load(stream, Loader=yaml.FullLoader)
        model = MUNIT_infer(hyperparameters)
        weights = torch.load(self.config['weights_path'])
        model.gen_a.load_state_dict(weights['a'])
        model.gen_b.load_state_dict(weights['b'])
        return model.to(self.device)

    def process_image(self, img_tensor):
        """Process single image"""
        # Create raindrop model
        drop_model = DropModel(
            imsize=self.config['imsize'],
            size_threshold=1/float(self.config['drop_size']),
            frequency_threshold=self.config['drop_frequency'],
            shape_threshold=self.config['drop_shape']
        )
        
        # Generate wetness effect
        im_wet = self.model.forward(img_tensor)
        
        # Add raindrop effect
        drops_sigma = torch.zeros(1).fill_(self.config['drop_sigma']).to(self.device)
        # Ensure im_wet size matches drop_model expected size
        if im_wet.size() != img_tensor.size():
            im_wet = transforms.Resize(img_tensor.shape[2:])(im_wet)
        im_rainy = drop_model.add_drops(im_wet, sigma=drops_sigma)
        
        # Convert back to PIL image
        im_wet = self.to_pil((im_wet[0].cpu() + 1) / 2)
        im_rainy = self.to_pil((im_rainy[0].cpu() + 1) / 2)
        
        return im_wet, im_rainy

def save_image(image, path):
    """Save image"""
    image.save(path)

def process_directory(input_dir, output_dir, config, resize=None):
    """Process all images in a directory"""
    # Check if CUDA device is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create processor
    processor = RainDropProcessor(config)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'wet'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rainy'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create progress bar using tqdm
    for img_file in tqdm(image_files, desc="Processing images", ncols=100):
        # Load image
        input_path = os.path.join(input_dir, img_file)
        img_tensor = load_image(input_path, resize=resize).to(device)
        
        # Process image
        wet_img, rainy_img = processor.process_image(img_tensor)
        
        # Save results
        output_name = os.path.splitext(img_file)[0]
        save_image(wet_img, os.path.join(output_dir, 'wet', f'{output_name}_wet.png'))
        save_image(rainy_img, os.path.join(output_dir, 'rainy', f'{output_name}_rainy.png'))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Raindrop effect generation program')
    parser.add_argument('--input_dir', default='./examples',
                      type=str, help='Input image directory')
    parser.add_argument('--output_dir', default='./output', type=str, help='Output image directory')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image resolution, default is 512x512')
    
    args = parser.parse_args()
    
    # Configure parameters
    config = {
        'drop_size': 50,
        'drop_frequency': 8,
        'drop_shape': 0.6,
        'drop_sigma': 4,
        'imsize': tuple(args.image_size),
        'params_path': './rainy/GuidedDisent/configs/params_net.yaml',
        'weights_path': './rainy/GuidedDisent/weights/pretrained.pth'
    }
    
    # Process directory
    process_directory(args.input_dir, args.output_dir, config, resize=config['imsize'])

if __name__ == '__main__':
    main()
