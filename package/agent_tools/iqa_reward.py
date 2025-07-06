
import os
import io
import numpy as np
import torch
import pyiqa
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# XXX: Set cache directory for transformers
os.environ['TRANSFORMERS_CACHE'] = 'xxx' #   

class IQAScore:
    """
    Image Quality Assessment class that implements multiple IQA metrics.
    
    This class provides methods to evaluate image quality using state-of-the-art
    IQA models like QAlign, MANIQA, MUSIQ, CLIPIQA, and NIQE.
    """
    
    def __init__(self, device='cuda', score_weight=None):
        """
        Initialize the IQA Score calculator.
        
        Args:
            device (str): Computing device ('cuda' or 'cpu').
            score_weight (list, optional): Weight for each metric [qalign, maniqa, musiq, clipiqa, niqe].
                                          Defaults to [1,1,1,1,1].
        """
        self.device = device
        self.score_weight = score_weight if score_weight is not None else [1, 1, 1, 1, 1]
        
        # Path to the QAlign model
        qalign_path = 'xxx'
        
        # Initialize metrics based on weights
        print("Loading IQA metrics...")
        
        # Only load models with non-zero weights to save memory
        if self.score_weight[0] != 0:
            self.qalign_metric = AutoModelForCausalLM.from_pretrained(
                qalign_path, trust_remote_code=True, 
                torch_dtype=torch.float16, device_map="auto"
            )
            
        if self.score_weight[1] != 0:
            self.maniqa_metric = pyiqa.create_metric('maniqa', device=self.device)
            
        if self.score_weight[2] != 0:
            self.musiq_metric = pyiqa.create_metric('musiq', device=self.device)
            
        if self.score_weight[3] != 0:
            self.clipiqa_metric = pyiqa.create_metric('clipiqa', device=self.device)
            
        if self.score_weight[4] != 0:
            self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)
        
        # Mean and standard deviation for score normalization
        self.mean_std = {
            'maniqa': {'mean': 0.2092878860158957, 'std': 0.1245655639250264}, 
            'musiq': {'mean': 40.525839667740804, 'std': 16.94055156979329}, 
            'qalign': {'mean': 0.25762742254801686, 'std': 0.1830440687605635}
        }
        
        print("IQA metrics loaded successfully")
    
    def imread2tensor(self, img_source, rgb=False):
        """
        Convert various image sources to PIL Image.
        
        Args:
            img_source: Can be bytes, file path string, or PIL Image.
            rgb (bool): Whether to convert to RGB.
            
        Returns:
            PIL.Image: Loaded image.
            
        Raises:
            Exception: If the source type is not supported.
        """
        if isinstance(img_source, bytes):
            img = Image.open(io.BytesIO(img_source))
        elif isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise Exception("Unsupported source type")
            
        if rgb:
            img = img.convert('RGB')
            
        return img

    def preprocess_image(self, source_image_path):
        """
        Preprocess the image for quality assessment.
        
        Args:
            source_image_path (str or bytes or PIL.Image): Image to preprocess.
            
        Returns:
            PIL.Image: Preprocessed image.
        """
        return self.imread2tensor(source_image_path)

    def normalize_iqa_score(self, score, metric):
        """
        Normalize an IQA score using pre-computed mean and standard deviation.
        
        Args:
            score (float): Raw score.
            metric (str): Metric name ('maniqa', 'musiq', or 'qalign').
            
        Returns:
            float: Normalized score (z-score).
        """
        mean = self.mean_std[metric]['mean']
        std = self.mean_std[metric]['std']
        return (score - mean) / std

    def get_iqa_score(self, source_image_path, eval=False, is_score_weight=False):
        """
        Calculate quality scores for an image using multiple metrics.
        
        Args:
            source_image_path (str): Path to the image.
            eval (bool): Whether in evaluation mode.
            is_score_weight (bool): Whether to use weighted scoring.
            
        Returns:
            list or dict: List of scores [qalign, maniqa, musiq, clipiqa, niqe] or
                         dict with combined score and individual scores if is_score_weight=True.
        """
        if is_score_weight:
            return self.get_iqa_score_with_weight(source_image_path)
            
        with torch.no_grad():
            source_image = self.preprocess_image(source_image_path)
            
            # Calculate scores using each metric
            qalign_score = self.qalign_metric.score([source_image], task_="quality", input_="image").item()
            maniqa_score = self.maniqa_metric(source_image).item()
            musiq_score = self.musiq_metric(source_image).item()
            clipiqa_score = self.clipiqa_metric(source_image).item()
            niqe_score = self.niqe_metric(source_image).item()
            
            return [qalign_score, maniqa_score, musiq_score, clipiqa_score, niqe_score]

    def get_iqa_score_with_weight(self, source_image_path):
        """
        Calculate a weighted quality score using multiple metrics.
        
        Args:
            source_image_path (str): Path to the image.
            
        Returns:
            dict: Dictionary with combined weighted score and list of individual scores.
        """
        with torch.no_grad():
            source_image = self.preprocess_image(source_image_path)
            score = 0
            score_list = [0, 0, 0, 0, 0]
            
            # Calculate scores for each metric with non-zero weight
            if self.score_weight[0] != 0:
                qalign_score = self.qalign_metric.score([source_image], task_="quality", input_="image").item()
                score += qalign_score * self.score_weight[0]
                score_list[0] = qalign_score
                
            if self.score_weight[1] != 0:
                maniqa_score = self.maniqa_metric(source_image).item()
                score += maniqa_score * self.score_weight[1]
                score_list[1] = maniqa_score
                
            if self.score_weight[2] != 0:
                musiq_score = self.musiq_metric(source_image).item()
                score += musiq_score * self.score_weight[2]
                score_list[2] = musiq_score
                
            if self.score_weight[3] != 0:
                clipiqa_score = self.clipiqa_metric(source_image).item()
                score += clipiqa_score * self.score_weight[3]
                score_list[3] = clipiqa_score
                
            if self.score_weight[4] != 0:
                niqe_score = self.niqe_metric(source_image).item()
                score += niqe_score * self.score_weight[4]
                score_list[4] = niqe_score
                
            return {
                'score': score,
                'score_list': score_list
            }
            
    def get_folder_scores(self, folder_path):
        """
        Calculate average scores for a folder of images.
        
        This method is useful for calibration and obtaining score distributions.
        
        Args:
            folder_path (str): Path to folder containing images.
            
        Returns:
            dict: Dictionary with mean and standard deviation for each metric.
        """
        # Predefined subfolder structure
        img_sub_folder = {
            "adcd": "night_driving/ADCD",
            "anno": "night_driving/Dark_Zurich_train_anon",
            "jinlong": "night_driving/nuscene_jinlong",
            "night": "night_driving/nighttime_100k_val_clean",
            "rainy": "rain_driving/ACDC",
            "raindrop": "raindrop_yeying/day/samples",
            "fog": "fog_driving/Foggy_Zurich"
        }
        
        # Initialize score lists
        maniqa_scores = []
        musiq_scores = []
        qinstruct_scores = []
        
        # Process images with progress bar
        count = 0
        progress_bar = tqdm(total=10000, desc="Processing images")
        
        # Traverse each subfolder
        for subfolder_name, subfolder_path in img_sub_folder.items():
            full_subfolder_path = os.path.join(folder_path, subfolder_path)
            
            # Process each image in the subfolder
            for root, dirs, files in os.walk(full_subfolder_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(root, filename)
                        
                        # Calculate scores for the image
                        with torch.no_grad():
                            source_image = self.preprocess_image(img_path)
                            maniqa = self.maniqa_metric(source_image).item()
                            musiq = self.musiq_metric(source_image).item()
                            qalign = self.qalign_metric.score([source_image], task_="quality", input_="image").item()
                        
                        # Store scores
                        maniqa_scores.append(maniqa)
                        musiq_scores.append(musiq)
                        qinstruct_scores.append(qalign)
                        
                        # Update progress
                        count += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix({"Count": count})
        
        progress_bar.close()

        # Calculate statistics
        return {
            'maniqa': {
                'mean': np.mean(maniqa_scores),
                'std': np.std(maniqa_scores)
            },
            'musiq': {
                'mean': np.mean(musiq_scores),
                'std': np.std(musiq_scores)
            },
            'qalign': {
                'mean': np.mean(qinstruct_scores),
                'std': np.std(qinstruct_scores)
            }
        }