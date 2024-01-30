import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria, TextStreamer

from PIL import Image
import random
import math

import requests
from PIL import Image
from io import BytesIO
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def image_parser(image_file):
    out = image_file.split(sep=',')
    return out


class LLAVA():
    def __init__(self, model_path="liuhaotian/llava-v1.5-13b", mm_projector_setting=None,
                     vision_tower_setting=None, conv_mode='vicuna_v1', temperature=0.0, CACHE="/data/tir/projects/tir6/general/sakter/cache"):
        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )

        self.mm_use_im_start_end = model.config.mm_use_im_start_end
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model
        self.temperature = temperature
        self.model_type = 'llava'

    def decode_output_text(self, output_ids, input_ids, conv, stop_str):
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )

        final_out = []
        for out in outputs:
            out = out.strip()
            if out.endswith(stop_str):
                out = out[: -len(stop_str)]
            out = out.strip()
            final_out.append(out)

        return final_out

    def ask(self, img_path, text, system_prompt=None, max_gen=1):
        qs = text
        if img_path is not None:
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs


        if self.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'

        conv = conv_templates[self.conv_mode].copy()
        if system_prompt:
            conv.system = system_prompt
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if img_path is None:
            images_tensor = None
        else:
            image_files = image_parser(img_path)
            images = load_images(image_files)
            images_tensor = process_images(
                images,
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=1.0,
                num_beams=1.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                num_return_sequences=max_gen,
            )

        outputs = self.decode_output_text(output_ids, input_ids, conv=conv, stop_str=stop_str)

        return outputs

    def caption(self, img_path):
        return self.ask(img_path=img_path, text='Give a clear and concise summary of the image below in one paragraph.')


if __name__ == "__main__":
    llava_model = LLAVA()
    print('successfully initialized llava model')
