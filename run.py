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

import prompts
from prompts import html, basic
import utils
from utils.utils import *
    


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
@click.option("--method", default='with_image', type=str)
@click.option("--sc", is_flag=True)
@click.option("--temp", default=0.0, type=float)
@click.option("--max_gen", default=1, type=int)
def main(lr, rr, task, method, temp, max_gen, sc):
    
    question_image_dict = {}
    
    if task == 'gsm8k':
        test_examples = get_examples_gsm8k(task, lr=lr, rr=rr)
    elif task in ['svamp', 'asdiv', 'mawpsmultiarith']:
        test_examples = get_examples_svamp(task, lr=lr, rr=rr)
    else:
        test_examples = get_examples(task, lr=lr, rr=rr)

        
    test_dset = GSMDataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    vqa_model = LLAVA(temperature=temp, conv_mode='llava_v0')
    
    all_responses = []
    
    if sc:
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}')
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}/{method}'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}/{method}')
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}/{method}/all_jsons'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/self_consistency/{task}/{method}/all_jsons')
    else:
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/{task}'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/{task}')
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/{task}/{method}'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/{task}/{method}')
        if not os.path.exists(f'/home/sakter/courses/Fall_2023/outputs/{task}/{method}/all_jsons'):
            os.makedirs(f'/home/sakter/courses/Fall_2023/outputs/{task}/{method}/all_jsons')

    
    path = f'/home/sakter/courses/Fall_2023/outputs/{task}/{method}'
    if sc:
        path = f'outputs/self_consistency/{task}/{method}'
    
    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        for qi, q, a in zip(qid, qn, ans):
            
            if method == 'with_image':
                img_path = f'/home/sakter/courses/Fall_2023/images/{task}_html/{qi}.jpg'
            else: 
                img_path = '/home/sakter/courses/Fall_2023/images/dummy_img.png'
            
            result_path = f'{path}/all_jsons/{qi}.json'
            
            assert os.path.isfile(img_path)
            if os.path.isfile(result_path):
                continue
                
            if task in ['asdiv', 'svamp', 'gsm8k', 'mawpsmultiarith']: q_prompt = basic.TASK_PROMPT['maths'][method.upper()].format(question=q)
            else: q_prompt = basic.TASK_PROMPT[task][method.upper()].format(question=q)

            
            response = vqa_model.ask(img_path, q_prompt, max_gen=max_gen)
            al = {'qid': qi.item(), 
                  'prompt': q_prompt, 
                  'question': q,
                  'predict': response, 
                  'answer': a}
            all_responses.append(al)
            with open(result_path, 'w') as fp:
                json.dump(al, fp)
    
    if task in ['asdiv', 'svamp', 'gsm8k', 'mawpsmultiarith']: all_responses = return_predicted_answer(all_responses)
    else: all_responses = get_answer(all_responses)
    
    with open(f'{path}/output.jsonl', 'w') as f:
        for d in all_responses:
            json.dump(d, f)
            f.write('\n')
        
    return


if __name__ == "__main__":
    main()