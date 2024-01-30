import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

import json
import click
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as onp
from PIL import Image
from einops import rearrange, reduce, repeat, pack, unpack

from accelerate import Accelerator
from models.math_llava import LLAVA
from models.gemini import Gemini

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader

import prompts
from prompts import html, basic
import utils
from utils.utils import *

class QADataset(Dataset):
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


    test_dset = QADataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    vqa_model = LLAVA(temperature=temp, conv_mode='llava_v0')

    all_responses = []

    if sc:
        Path(f'outputs/self_consistency/{task}').mkdir(parents=True, exist_ok=True)
        Path(f'outputs/self_consistency/{task}/{method}').mkdir(parents=True, exist_ok=True)
        Path(f'outputs/self_consistency/{task}/{method}/all_jsons').mkdir(parents=True, exist_ok=True)
        path = Path(f"outputs/self_consistency/{task}/{method}")
    else:
        Path(f'outputs/{task}').mkdir(parents=True, exist_ok=True)
        Path(f'outputs/{task}/{method}').mkdir(parents=True, exist_ok=True)
        Path(f'outputs/{task}/{method}/all_jsons').mkdir(parents=True, exist_ok=True)
        path = Path(f"outputs/{task}/{method}")

    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        for qi, q, a in zip(qid, qn, ans):
            if method == 'with_image':
                img_path = Path(f'images/{task}_html/{qi}.jpg')
            elif method == 'without_image':
                img_path = Path('images/dummy.png')
            else:
                raise NotImplementedError

            result_path = path / f'all_jsons/{qi}.json'

            if result_path.exists():
                continue

            if task in ['asdiv', 'svamp', 'gsm8k', 'mawpsmultiarith']:
                task = 'maths'
            q_prompt = basic.TASK_PROMPT[task][method.upper()].format(question=q)

            response = vqa_model.ask(img_path, q_prompt, max_gen=max_gen)

            al = {
                'qid': qi.item(),
                'prompt': q_prompt,
                'question': q,
                'predict': response,
                'answer': a
            }

            all_responses.append(al)

            with open(result_path, 'w') as fp:
                json.dump(al, fp)

    if task in ['asdiv', 'svamp', 'gsm8k', 'mawpsmultiarith']:
        all_responses = return_predicted_answer(all_responses)
    else:
        all_responses = get_answer(all_responses)

    with open(path / 'output.jsonl', 'w') as f:
        for d in all_responses:
            json.dump(d, f)
            f.write('\n')


if __name__ == "__main__":
    main()
