<div align="center">
<div align="center">
  <img src="assets/icon.png" alt="JarvisIR Logo" width="100px">
  <h1>[CVPR' 2025] JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration</h1>
</div>

<a href="https://lyl1015.github.io/papers/CVPR2025_JarvisIR.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-JarvisIR-b31b1b" alt="Paper PDF">
</a>
<!-- <a href="#"><img src="https://img.shields.io/badge/arXiv-即将发布-b31b1b" alt="arXiv"></a> -->
<a href="https://cvpr2025-jarvisir.github.io/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
<a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Coming%20Soon-blue" alt="Demo"></a>
<a href="https://github.com/LYL1015/JarvisIR?tab=readme-ov-file/"><img src="https://img.shields.io/badge/GitHub-Code-black" alt="Code"></a>

<!-- **提升自动驾驶感知能力的智能图像恢复技术** -->

<!-- **[厦门大学](https://www.xmu.edu.cn/)** | **[香港科技大学（广州）](https://hkust-gz.edu.cn/)** | **[字节跳动Pico](https://www.picoxr.com/)** | **[腾讯](https://www.tencent.com/)** | **[华为诺亚方舟实验室](https://www.huawei.com/)** | **[香港中文大学](https://www.cuhk.edu.hk/)** -->


[Yunlong Lin](https://lyl1015.github.io/)<sup>1*♣</sup>, [Zixu Lin](https://github.com/)<sup>1*♣</sup>, [Haoyu Chen](https://haoyuchen.com/)<sup>2*</sup>, [Panwang Pan](https://paulpanwang.github.io/)<sup>3*</sup>, [Chenxin Li](https://chenxinli001.github.io/)<sup>6</sup>, [Sixiang Chen](https://ephemeral182.github.io/)<sup>2</sup>, [Kairun Wen](https://kairunwen.github.io/)<sup>1</sup>, [Yeying Jin](https://jinyeying.github.io/)<sup>4</sup>, [Wenbo Li](https://fenglinglwb.github.io/)<sup>5†</sup>, [Xinghao Ding](https://scholar.google.com/citations?user=k5hVBfMAAAAJ&hl=zh-CN)<sup>1†</sup>

<sup>1</sup>Xiamen University, <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou), <sup>3</sup>Bytedance's Pico, <sup>4</sup>Tencent, <sup>5</sup>Huawei Noah's Ark Lab, <sup>6</sup>The Chinese University of Hong Kong
<!-- <sup>*</sup>Equal Contribution <sup>♣</sup>Equal Contribution <sup>†</sup>Corresponding Author -->
Accepted by CVPR 2025
</div>


## :postbox: Updates
<!-- - 2023.12.04: Add an option to speed up the inference process by adjusting the number of denoising steps. -->
<!-- - 2024.2.9: Release our demo codes and models. Have fun! :yum: -->
- 2025.4.8: This repo is created.

## :diamonds: Overview
JarvisIR (CVPR 2025) is a VLM-powered agent designed to tackle the challenges of vision-centric perception systems under unpredictable and coupled weather degradations. It leverages the VLM as a controller to manage multiple expert restoration models, enabling robust and autonomous operation in real-world conditions. JarvisIR employs a novel two-stage framework consisting of supervised fine-tuning and human feedback alignment, allowing it to effectively fine-tune on large-scale real-world data in an unsupervised manner. Supported by CleanBench, a comprehensive dataset with 150K synthetic and 80K real instruction-response pairs, JarvisIR demonstrates superior decision-making and restoration capabilities, achieving a 50% improvement in the average of all perception metrics on CleanBench-Real.
<div align="center">
  <img src="assets/teaser1.png" alt="JarvisIR Teaser" width="800px">
</div>


## :circus_tent: Checklist

- [ ] Release inference code
- [ ] Release Hugging Face demo
- [ ] Release CleanBench data
- [ ] Release training code

## :love_you_gesture: Citation
```bibtex
@inproceedings{jarvisir2025,
  title={JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration},
  author={Lin, Yunlong and Lin, Zixu and Chen, Haoyu and Pan, Panwang and Li, Chenxin and Chen, Sixiang and Jin, Yeying and Li, Wenbo and Ding, Xinghao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```