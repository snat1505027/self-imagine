import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, TaskType, LoraConfig
from peft import prepare_model_for_kbit_training

import json
import click
import wandb
import deeplake
from pathlib import Path
from tqdm import tqdm
import numpy as onp
from PIL import Image
from einops import rearrange, reduce, repeat, pack, unpack


from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup, Blip2VisionConfig, Blip2QFormerConfig, OPTConfig, Blip2Config, Blip2Processor, Blip2ForConditionalGeneration
from transformers import BitsAndBytesConfig
from math_llava import LLAVA

import os
import re
import torch as th
import imgkit

from prompts import html, basic


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.ids = [ex["qid"] for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn = self.qns[idx]
        ans = self.ans[idx]
        qid = self.ids[idx]
        
        return qid, qn, ans

@click.command()
@click.option("--lr", default=0, type=int)
@click.option("--rr", default=500, type=int)
@click.option("--task", default='gsm8k', type=str)
def main(lr, rr, task):
    
    question_image_dict = {}
    
    if task == 'gsm8k':
        test_examples = get_examples_gsm8k("test_qid", lr=lr, rr=rr)
    elif task in ['svamp', 'asdiv', 'mawpsmultiarith']:
        test_examples = get_examples_svamp(task, lr=lr, rr=rr)
    else:
        test_examples = get_examples(task, lr=lr, rr=rr)
        
        
        
    test_dset = GSMDataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    vqa_model = LLAVA(temperature=0.0)
    
    all_responses = {}
    img_path = f'/home/sakter/courses/Fall_2023/images/dummy_img.png'
    
    
    if not os.path.exists(f'images/{task}'):
        os.makedirs(f'images/{task}')
        
    if not os.path.exists(f'htmls/{task.upper()}'):
        os.makedirs(f'htmls/{task.upper()}')
    
    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        for qi, q, a in zip(qid, qn, ans):
            
            result_path = f'/home/sakter/courses/Fall_2023/htmls/{task.upper()}_html/math_{qi}.json'
            
            assert os.path.isfile(img_path)
            if os.path.isfile(result_path):
                continue
            q_prompt = html_prompts.HTML_PROMPT.format(question=q)
            
            response = vqa_model.ask(img_path, q_prompt)
            all_responses[q] = {'qid': qi.item(), 'question': q_prompt, 'predict': response, 'answer': a}
            with open(result_path, 'w') as fp:
                json.dump(all_responses[q], fp)
            
            soln = response.split('Q:')[0].strip()
            img_path = f'/home/sakter/courses/Fall_2023/images/{task}/{qi.item()}.jpg'
            if os.path.isfile(img_path):
                continue
            with open("temp_result.html", "w") as f:
                f.write(soln)
            imgkit.from_file('temp_result.html', img_path, options={"quality":100, 'width': 400, 'disable-smart-width': '' })
        

      
    return


if __name__ == "__main__":
    main()