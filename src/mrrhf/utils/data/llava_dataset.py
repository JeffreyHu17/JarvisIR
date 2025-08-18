# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://llava-vl.github.io/
import os
import copy
import utils.data.DST as DST
from .vqa_dataset import VQADataset
from utils.utils import get_rank
from .utils import save_debug_text


import torch
import numpy as np


class LlavaPPODataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        per_sample_image = 1
        self.img_info = kwargs.pop("img_info", None)
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, add_eos=False, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
        if (res["input_ids"][-1] == self.tokenizer.eos_token_id) and (not self.add_eos):
            res["input_ids"] = res["input_ids"][0:-1]
            res["attention_mask"] = res["attention_mask"][0:-1]

        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :] # lkj：截取指令部分，后面部分在计算损失时忽略

        res.update(labels=labels)
        return res

    def __getitem__(self, index):
        res_list = []
        for ann in self.annotation[index]:
            if ann['image'] != None:
                if self.template == 'llava_next':
                    image, image_sizes = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
                else:
                    image, img_path = self.process_image(ann,
                                            data_debug_path=self.data_debug_path,
                                            data_debug_counter=self.data_debug_counter)
                with_image = True
            else:
                image = None
                with_image = False
            
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=True,
                                    with_image=with_image)

            self.data_debug_counter += 1
            res = self.tokenize(text)
            res.update(image=image)
            res.update(text)
            res["img_path"] = img_path
            res_list.append(res)
        
        output = self.merge_all_images(res_list, text)
        return output

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, with_image=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        # question = question.replace("<image>", "").replace("\n", "")
        question = question.replace("<image>", "").strip("\n")

        answer = ann["conversations"][1]["value"] 
        query1 = ann["conversations"][0]["value"] 
        instruction = self.prompter(query1, with_image=True, first_message=True, template=self.template)

        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)
    