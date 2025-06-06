import os
os.environ["BASICSR_JIT"] = "True"
from PIL import Image
import torch
from .SCUNet.inference import load_scu_model, scu_predict
from .Retinexformer.inference import load_retinexformer_model, retinexformer_predict
from .img2img_turbo.inference import load_turbo_model, turbo_predict
from .ESRGAN.inference import load_esrgan_model, esrgan_predict
from .RIDCP.inference import load_ridcp_model, ridcp_predict
from .LightenDiffusion.inference import load_lightdiff_model, lightdiff_predict
from .IDT.inference import load_idt_model, idt_predict
from .iqa_reward import IQAScore
from .SnowMaster.inference import load_snowmaster_model, snowmaster_predict
from .S2Former.inference import load_s2former_model, s2former_predict
from .KANet.inference import load_kanet_model, kanet_predict
from .HVICIDNet.inference import load_hvicidnet_model, hvicidnet_predict

class RestorationToolkit():
    """
    A toolkit for image restoration, providing access to various models and evaluation capabilities.
    """
    def __init__(self, models=None, device='cuda', score_weight=None):
        """
        Initialize the toolkit engine.
        
        Args:
            models (list, optional): A list of models to load. Defaults to all available models.
            device (str, optional): The computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
            score_weight (dict, optional): Weights for IQA score calculation. Defaults to None.
        """
        print("Loading models...")
        self.all_model_paths = [
            'scunet', 
            'retinexformer_fivek', 'hvicidnet', 'lightdiff',
            'turbo_rain', 'idt', 's2former',
            'ridcp', 'kanet', 
            'turbo_snow', 'snowmaster',
            'real_esrgan',
        ]
        
        if models is not None:
            self.all_model_paths = models
            
        # Model file paths configuration
        self.model_paths = {
            'scunet': os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../checkpoints/agent_tools/SCUNet/scunet_color_real_gan.pth'),
            'retinexformer': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/Retinexformer'),
            'real_esrgan': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/ESRGAN/RealESRGAN_x4plus.pth'),
            'ridcp': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/RIDCP'),
            'idt': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/IDT'),
            'img2img_turbo': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/img2img_turbo'),
            'lightdiff': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/LightenDiffusion'),
            'hvicidnet': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/HVICIDNet/generalization.pth'),
            's2former': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/S2Former'),
            'kanet': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/KANet'),
            'snowmaster': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/SnowMaster'),
            'turbo': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../checkpoints/agent_tools/Img2img_turbo')
        }
        
        self.device = device
        self.models = {}
        self.load_models()
        self.iqa = IQAScore(self.device, score_weight)
        print(f"Finished loading models. models: {self.models.keys()}")

    def load_models(self):
        """
        Load all configured models.
        """
        for model_name in self.all_model_paths:
            if model_name == 'scunet':
                self.models['scunet'] = load_scu_model(self.model_paths[model_name], self.device)
            elif model_name == 'retinexformer_fivek':
                self.models['retinexformer_fivek'] = load_retinexformer_model(self.model_paths['retinexformer'], self.device)
            elif model_name == 'turbo_rain':
                self.models['turbo_rain'] = load_turbo_model('rain', self.model_paths['turbo'], self.device)
            elif model_name == 'turbo_snow':
                self.models['turbo_snow'] = load_turbo_model('snow', self.model_paths['turbo'], self.device)
            elif model_name == 'real_esrgan':
               self.models['real_esrgan'] = load_esrgan_model(self.model_paths[model_name], self.device)
            elif model_name == 'ridcp':
                self.models['ridcp'] = load_ridcp_model(self.model_paths[model_name], self.device)
            elif model_name == 'idt':
                self.models['idt'] = load_idt_model('day', self.model_paths['idt'], self.device)
            elif model_name == 'lightdiff':
                self.models['lightdiff'] = load_lightdiff_model(self.model_paths[model_name], self.device)
            elif model_name == 'snowmaster':
                self.models['snowmaster'] = load_snowmaster_model(self.model_paths[model_name], self.device)
            elif model_name == 's2former':
                self.models['s2former'] = load_s2former_model(self.model_paths[model_name], self.device)
            elif model_name == 'kanet':
                self.models['kanet'] = load_kanet_model(self.model_paths[model_name], self.device)
            elif model_name == 'hvicidnet':
                self.models['hvicidnet'] = load_hvicidnet_model(self.model_paths[model_name], self.device)

    def resize_image(self, img_path, output_dir):
        """
        Resize image to 512x512.
        
        Args:
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the resized image.
            
        Returns:
            str: Path to the resized image.
        """
        with Image.open(img_path) as img:
            img = img.convert('RGB')  # Ensure consistent color mode
            img = img.resize((512, 512), Image.LANCZOS)  # Use high-quality resampling
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{img_name}.png")
            img.save(save_path, format='PNG')
        return save_path

    def process_image_with_models(self, model_list, img_path, output_dir):
        """
        Process an image with a specified sequence of models.
        
        Args:
            model_list (list): A list of model names to use for processing.
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the output images.
            
        Returns:
            str: The absolute path to the final processed image.
        """
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            img_path = self.resize_image(img_path, output_dir)
            for model_name in model_list:
                if model_name not in self.all_model_paths:
                    print(f"Model {model_name} not found in available models")
                    continue
                if model_name == 'scunet':
                    img_path = scu_predict(self.models['scunet'], img_path, output_dir, device=self.device)
                elif model_name == 'retinexformer_fivek':
                    img_path = retinexformer_predict(self.models['retinexformer_fivek'], img_path, output_dir, device=self.device)
                elif model_name == 'turbo_rain':
                    img_path = turbo_predict(self.models['turbo_rain'], img_path, output_dir, device=self.device)
                elif model_name == 'turbo_snow':
                    img_path = turbo_predict(self.models['turbo_snow'], img_path, output_dir, device=self.device)
                elif model_name == 'real_esrgan':
                    img_path = esrgan_predict(self.models['real_esrgan'], img_path, output_dir, device=self.device)
                elif model_name == 'ridcp':
                    img_path = ridcp_predict(self.models['ridcp'], img_path, output_dir, device=self.device)
                elif model_name == 'idt':
                    img_path = idt_predict(self.models['idt'], img_path, output_dir, device=self.device)
                elif model_name == 'lightdiff':
                    img_path = lightdiff_predict(self.models['lightdiff'], img_path, output_dir, device=self.device)
                elif model_name == 'snowmaster':
                    img_path = snowmaster_predict(self.models['snowmaster'], img_path, output_dir, device=self.device)
                elif model_name == 's2former':
                    img_path = s2former_predict(self.models['s2former'], img_path, output_dir, device=self.device)
                elif model_name == 'kanet':
                    img_path = kanet_predict(self.models['kanet'], img_path, output_dir, device=self.device)
                elif model_name == 'hvicidnet':
                    img_path = hvicidnet_predict(self.models['hvicidnet'], img_path, output_dir, device=self.device)
        return os.path.abspath(img_path)  # Return the absolute path to the final image
    
    def process_image(self, tools, img_path, output_dir, eval=False, is_score_weight=False):
        """
        Process an image and calculate its quality score.
        
        Args:
            tools (list): A list of tool names to use for processing.
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the output image.
            eval (bool, optional): Whether to perform evaluation. Defaults to False.
            is_score_weight (bool, optional): Whether to use weights for scoring. Defaults to False.
            
        Returns:
            dict: A dictionary containing the output path and the calculated score.
        """
        # Call the processing function
        output_path = self.process_image_with_models(tools, img_path, output_dir)
        # score = self.iqa.get_iqa_score(output_path, eval, is_score_weight)
        score = [0,0,0,0,0]
        return {"output_path": output_path, "score": score}