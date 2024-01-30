import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import json
import click
from pathlib import Path
from tqdm import tqdm
import numpy as onp
from PIL import Image

from models.math_llava import LLAVA
from models.gemini import Gemini
from utils.utils import *

import imgkit
from prompts import html, basic
from dotenv import load_dotenv
load_dotenv()

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
@click.option("--model", default='llava', type=str)
def main(lr, rr, task, model):
    if task == 'gsm8k':
        test_examples = get_examples_gsm8k("test_qid", lr=lr, rr=rr)
    elif task in ['svamp', 'asdiv', 'mawpsmultiarith']:
        test_examples = get_examples_svamp(task, lr=lr, rr=rr)
    else:
        test_examples = get_examples(task, lr=lr, rr=rr)

    test_dset = QADataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    if model == 'gemini':
        vqa_model = Gemini(temperature=0.0)
    elif model == 'llava':
        vqa_model = LLAVA(temperature=0.0)
    else:
        raise NotImplementedError

    all_responses = {}
    img_path = f'images/dummy.png'

    Path(f'images/{task}').mkdir(parents=True, exist_ok=True)
    Path(f'htmls/{task.upper()}').mkdir(parents=True, exist_ok=True)

    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):

        for qi, q, a in zip(qid, qn, ans):

            result_path = Path(f'htmls/{task.upper()}_html/math_{qi}.json')
            if result_path.exists():
                continue

            q_prompt = html.HTML_PROMPT.format(question=q)

            response = vqa_model.ask(img_path, q_prompt)
            all_responses[q] = {'qid': qi.item(), 'question': q_prompt, 'predict': response, 'answer': a}
            with open(result_path, 'w') as fp:
                json.dump(all_responses[q], fp)

            soln = response.split('Q:')[0].strip()
            img_path = Path(f'images/{task}/{qi.item()}.jpg')

            if img_path.exists():
                continue

            with open("temp_result.html", "w") as f:
                f.write(soln)
            imgkit.from_file('temp_result.html', img_path, options={"quality":100, 'width': 400, 'disable-smart-width': '' })

if __name__ == "__main__":
    main()
