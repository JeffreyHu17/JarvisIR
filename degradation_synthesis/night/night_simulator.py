import torchvision
from torchvision import transforms
import torch
import numpy as np
import random
from scipy import stats
import os
def save_image(tensor, path):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(path)

def random_noise_levels(device):
    """Generates random shot and read noise from a log-log linear distribution."""
    # log_min_shot_noise = torch.log(torch.tensor(0.0001, device=device))
    # log_max_shot_noise = torch.log(torch.tensor(0.012, device=device))
    log_min_shot_noise = torch.log(torch.tensor(0.000005, device=device))
    log_max_shot_noise = torch.log(torch.tensor(0.003, device=device))


    log_shot_noise = torch.rand(1, device=device) * (log_max_shot_noise - log_min_shot_noise) + log_min_shot_noise
    shot_noise = torch.exp(log_shot_noise)

    # line = lambda x: 2.18 * x + 1.20
    line = lambda x: 1 * x + 0.05  # 减小斜率和截距

    log_read_noise = line(log_shot_noise) + torch.randn(1, device=device) * 0.03
    read_noise = torch.exp(log_read_noise)
    return shot_noise, read_noise

class ImageProcessor:
    def __init__(self, degration_cfg):
        self.degration_cfg = degration_cfg

    def apply_ccm(self, image, ccm):
        """
        Apply the CCM matrix.
        """
        image = image.float()  # Cast image to float32
        ccm = ccm.float()      # Cast ccm to float32
        shape = image.shape
        image = image.reshape(-1, 3)  # Replace view with reshape
        image = torch.matmul(image, ccm.T)  # T for transpose
        return image.view(shape)

    def Low_Illumination_Degrading(self, imgs, img_meta=None, safe_invert=False):
        """
        Degrades a batch of low-light images.
        img: Tensor of shape (B, C, H, W), where B is the batch size.
        """
        device = imgs.device
        config = self.degration_cfg
        B, C, H, W = imgs.shape  # Batch size, channels, height, width
        
        # Camera color matrix
        xyz2cams = torch.tensor([[[1.0234, -0.2969, -0.2266],
                                  [-0.5625, 1.6328, -0.0469],
                                  [-0.0703, 0.2188, 0.6406]],
                                 [[0.4913, -0.0541, -0.0202],
                                  [-0.613, 1.3513, 0.2906],
                                  [-0.1564, 0.2151, 0.7183]],
                                 [[0.838, -0.263, -0.0639],
                                  [-0.2887, 1.0725, 0.2496],
                                  [-0.0627, 0.1427, 0.5438]],
                                 [[0.6596, -0.2079, -0.0562],
                                  [-0.4782, 1.3016, 0.1933],
                                  [-0.097, 0.1581, 0.5181]]], device=device)
        rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]], device=device)
        
        # Unprocess part (RGB to RAW)
        imgs = imgs.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Inverse tone mapping
        imgs = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * imgs) / 3.0)
        
        # Inverse gamma
        epsilon = torch.tensor([1e-8], device=device)
        gamma = torch.rand(B, 1, 1, 1, device=device) * (config['gamma_range'][1] - config['gamma_range'][0]) + config['gamma_range'][0]
        imgs = torch.max(imgs, epsilon) ** gamma

        # sRGB to cRGB
        idx = random.randint(0, xyz2cams.shape[0] - 1)
        xyz2cam = xyz2cams[idx]
        rgb2cam = torch.matmul(xyz2cam, rgb2xyz)
        rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True)
        imgs = self.apply_ccm(imgs, rgb2cam)

        # Inverse white balance
        rgb_gain = torch.normal(mean=config['rgb_range'][0], std=config['rgb_range'][1], size=(B, 1, 1, 1), device=device)
        red_gain = torch.rand(B, 1, 1, 1, device=device) * (config['red_range'][1] - config['red_range'][0]) + config['red_range'][0]
        blue_gain = torch.rand(B, 1, 1, 1, device=device) * (config['blue_range'][1] - config['blue_range'][0]) + config['blue_range'][0]

        gains1 = torch.cat([1.0 / red_gain, torch.ones_like(red_gain), 1.0 / blue_gain], dim=-1) * rgb_gain

        if safe_invert:
            img_gray = torch.mean(imgs, dim=-1, keepdim=True)
            inflection = 0.9
            mask = (torch.max(img_gray - inflection, torch.zeros_like(img_gray)) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
            imgs = torch.clamp(imgs * safe_gains, min=0.0, max=1.0)
        else:
            imgs = imgs * gains1

        # Darkness (low photon numbers)
        lower, upper = config['darkness_range']
        # mu, sigma = 0.1, 0.08
        mu, sigma = 0.06, 0.04

        darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(size=B)
        darkness = torch.tensor(darkness, device=device).view(B, 1, 1, 1)
        imgs = imgs * darkness
        
        # Add shot and read noise
        shot_noise, read_noise = random_noise_levels(device)
        var = imgs * shot_noise*0.1  + read_noise*0.1
        var = torch.max(var, epsilon)
        noise = torch.normal(mean=0, std=torch.sqrt(var)).to(device)
        imgs = imgs + noise

        # ISP part (RAW to RGB)
        bits = random.choice(config['quantisation'])
        quan_noise = torch.FloatTensor(imgs.size()).uniform_(-1 / (255 * bits), 1 / (255 * bits)).to(device)
        imgs = imgs + quan_noise

        # White balance
        gains2 = torch.cat([red_gain, torch.ones_like(red_gain), blue_gain], dim=-1)
        imgs = imgs * gains2

        # cRGB to sRGB
        cam2rgb = torch.inverse(rgb2cam)
        imgs = self.apply_ccm(imgs, cam2rgb)

        # Gamma correction
        imgs = torch.max(imgs, epsilon) ** (1 / gamma)
        
        # Re-permute to (B, C, H, W)
        imgs = imgs.permute(0, 3, 1, 2)


        return imgs