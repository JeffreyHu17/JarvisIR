# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE


cmd_args = lambda: None
cmd_args.llm_name_or_path = None # Path to the LLM model
cmd_args.visual_encoder_name_or_path = None # Path to the visual encoder model
cmd_args.pretrained_pth = None # Path to the pretrained model weights
cmd_args.data_path = None # Path to the training data JSON file
cmd_args.image_folder = None # Path to the image folder
cmd_args.evaluation_images = None # Path to the evaluation images
cmd_args.max_epochs = None # Maximum number of training epochs
cmd_args.batch_size = None # Batch size per device
cmd_args.accumulative_counts = None # Gradient accumulation steps
cmd_args.dataloader_num_workers = None # Number of dataloader workers
cmd_args.lr = None # Learning rate
cmd_args.weight_decay = None # Weight decay
cmd_args.max_norm = None # Gradient clipping max norm
cmd_args.warmup_ratio = None # Warmup ratio
cmd_args.save_steps = None # Save checkpoint every X steps
cmd_args.save_total_limit = None # Maximum number of checkpoints to keep
cmd_args.evaluation_freq = None # Evaluation frequency in iterations


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model - Use command line args if provided, otherwise use defaults
llm_name_or_path = getattr(cmd_args, 'llm_name_or_path', None) or './checkpoints/base_models/llava-llama-3-8b'
visual_encoder_name_or_path = getattr(cmd_args, 'visual_encoder_name_or_path', None) or './checkpoints/base_models/clip-vit-large-patch14-336'
# Specify the pretrained pth
pretrained_pth = getattr(cmd_args, 'pretrained_pth', None) or './checkpoints/pretrained_models/preview'  # noqa: E501

# Data
data_path = getattr(cmd_args, 'data_path', None) or 'data/CleanBench/train/sft_data.json'
image_folder = getattr(cmd_args, 'image_folder', None) or 'data/CleanBench/train/images'
prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = int(2048 - (336 / 14) ** 2)

# Scheduler & Optimizer
batch_size = getattr(cmd_args, 'batch_size', None) or 16  # per_device
accumulative_counts = getattr(cmd_args, 'accumulative_counts', None) or 2
dataloader_num_workers = getattr(cmd_args, 'dataloader_num_workers', None) or 4
max_epochs = getattr(cmd_args, 'max_epochs', None) or 2
optim_type = AdamW
lr = getattr(cmd_args, 'lr', None) or 2e-5
betas = (0.9, 0.999)
weight_decay = getattr(cmd_args, 'weight_decay', None) or 0
max_norm = getattr(cmd_args, 'max_norm', None) or 1  # grad clip
warmup_ratio = getattr(cmd_args, 'warmup_ratio', None) or 0.03

# Save
save_steps = getattr(cmd_args, 'save_steps', None) or 1000
save_total_limit = getattr(cmd_args, 'save_total_limit', None) or 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = getattr(cmd_args, 'evaluation_freq', None) or 1000
SYSTEM = ""
evaluation_images = getattr(cmd_args, 'evaluation_images', None) or 'data/CleanBench/eval/images'

# Available image restoration tasks and their corresponding models
all_tasks = " {denoise: [scunet, restormer], lighten: [retinexformer_fivek, hvicidnet, lightdiff], \
                derain: [idt, turbo_rain, s2former], defog:[ridcp, kanet], \
                desnow:[turbo_snow, snowmaster], super_resolution: [real_esrgan], \
            }"

# Various prompt templates for querying the LLM about image degradation and restoration tasks
evaluation_inputs = [
    f"Considering the image's degradation, suggest the required tasks with explanations, and identify suitable tools for each task. Options for tasks and tools include: {all_tasks}.",
    f"Given the image's degradation, outline the essential tasks along with justifications, and choose the appropriate tools for each task from the following options: {all_tasks}.",
    f"Please specify the tasks required due to the image's degradation, explain the reasons, and select relevant tools for each task from the provided options: {all_tasks}.",
    f"Based on the image degradation, determine the necessary tasks and their reasons, along with the appropriate tools for each task. Choose from these options: {all_tasks}.",
    f"Identify the tasks required to address the image's degradation, including the reasons for each, and select tools from the options: {all_tasks}.",
    f"Considering the degradation observed, list the tasks needed and their justifications, then pick the most suitable tools for each task from these options: {all_tasks}.",
    f"Evaluate the image degradation, and based on that, provide the necessary tasks and reasons, along with tools chosen from the options: {all_tasks}.",
    f"With respect to the image degradation, outline the tasks needed and explain why, selecting tools from the following list: {all_tasks}.",
    f"Given the level of degradation in the image, specify tasks to address it, include reasons, and select tools for each task from: {all_tasks}.",
    f"Examine the image's degradation, propose relevant tasks and their explanations, and identify tools from the options provided: {all_tasks}.",
    f"Based on observed degradation, detail the tasks required, explain your choices, and select tools from these options: {all_tasks}.",
    f"Using the image's degradation as a guide, list the necessary tasks, include explanations, and pick tools from the provided choices: {all_tasks}.",
    f"Assess the image degradation, provide the essential tasks and reasons, and select the appropriate tools for each task from the options: {all_tasks}.",
    f"According to the image's degradation, determine which tasks are necessary and why, choosing tools for each task from: {all_tasks}.",
    f"Observe the degradation in the image, specify the needed tasks with justifications, and select appropriate tools from: {all_tasks}.",
    f"Taking the image degradation into account, specify tasks needed, provide reasons, and choose tools from the following: {all_tasks}.",
    f"Consider the image's degradation level, outline the tasks necessary, provide reasoning, and select suitable tools from: {all_tasks}.",
    f"Evaluate the degradation in the image, identify tasks required, explain your choices, and pick tools from: {all_tasks}.",
    f"Analyze the image degradation and suggest tasks with justifications, choosing the best tools from these options: {all_tasks}.",
    f"Review the image degradation, and based on it, specify tasks needed, provide reasons, and select tools for each task from: {all_tasks}."
]


# Display parsed command line arguments
if cmd_args is not None and any(vars(cmd_args).values()):
    print("\n===== 使用命令行参数 =====")
    for arg, value in vars(cmd_args).items():
        if value is not None:
            print(f"{arg}: {value}")
    print("=======================================\n")

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

model = dict(
    type=LLaVAModel,
    freeze_llm=False,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
    ),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=default_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)