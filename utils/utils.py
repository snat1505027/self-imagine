import os 
import shutil
import re
import json
from typing import Any, Callable, Iterable, Match, Optional, Pattern, Protocol, Sequence, Union


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
PATTERN = r"(?:\(|\s)([A-Z])\.?(?:\)|\s|$)"

tasks = ['date_understanding', 'geometric_shapes', 'navigate', 'temporal_sequences', 'tracking_shuffled_objects_three_objects',
        'tracking_shuffled_objects_seven_objects', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects']


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples_gsm8k(split, lr=0, rr=-1):
    path = os.path.join("/home/sakter/courses/Fall_2023/openai/data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    if rr == -1:
        examples = examples[lr:len(examples)]
    else:
        examples = examples[lr:rr]
        
    print(f"{len(examples)} {split} examples")
    return examples

def get_examples_svamp(split, lr=0, rr=-1):
    path = os.path.join("/home/sakter/courses/Fall_2023/openai/data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["input"] + "\n")
        ex.update(answer=str(ex["target"]))
        del ex['input']
        del ex["target"]

    if rr == -1:
        examples = examples[lr:len(examples)]
    else:
        examples = examples[lr:rr]
        
    print(f"{len(examples)} {split} examples")
    return examples

def get_examples(split, lr=0, rr=-1):
    path = os.path.join("/home/sakter/courses/Fall_2023/BIG-Bench-Hard/bbh/", f"{split}.jsonl")
    
    examples = read_jsonl(path)
        
    for ex in examples:
        ex.update(question=ex["input"]+ "\n")
        ex.update(answer=ex["target"])
        del ex['input']
        del ex["target"]

    if rr == -1:
        examples = examples[lr:len(examples)]
    else:
        examples = examples[lr:rr]
    print(f"{len(examples)} {split} examples")
    return examples
    
def find_numbers(x: str) -> list[str]:
    numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
      ).findall(x)
    return numbers


def find_number(x: str, answer_delimiter: Optional[str] = 'Answer:') -> str:
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ''


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def maybe_remove_comma(x: str) -> str:
    if is_float(x):
        return x
    return x.replace(',', '')

def return_predicted_answer(question_answer_list):
    correct = 0
    for out in question_answer_list:
        soln = out['response'].split('\nQ:')[0]
        short_responses = maybe_remove_comma(find_number(soln))
        
        if short_responses != '':
            correct += float(maybe_remove_comma(find_number(out['answer']))) == float(short_responses)
            out['is_correct'] = int(float(maybe_remove_comma(find_number(out['answer']))) == float(short_responses))
            out['predict'] = short_responses
        else:
            out['is_correct'] = 0
            out['predict'] = "-10000000000"
            
    print('Accuracy: ', correct/(1.0*len(question_answer_list)))
    return question_answer_list


def get_answer(question_answer_list):
    for out in question_answer_list:
        soln = out['response'].split('Q:')[0]
        exact = out['answer']
        prob_ans = re.findall(r"(?<=The answer is )(.*)", soln)
        prob_ans = [x[:-1] if x[-1] == '.' else x for x in prob_ans]
        if len(prob_ans) > 0 and exact == prob_ans[-1]:
            out['predict'] = out['answer']
            out['is_correct'] = 1
        else:
            if len(prob_ans) > 0: out['predict'] = str(prob_ans[-1])
            else: out['predict'] = "-10000000000"
            out['is_correct'] = 0
            
        
        if exact.startswith('(') and exact.endswith(')'):
            prob_ans = re.findall(PATTERN, out['response'])
            prob_ans = ['('+x+')' for x in prob_ans]
            if len(prob_ans) > 0 and exact == prob_ans[-1]:
                out['predict_last'] = out['answer']
                out['is_correct_last'] = 1
            else:
                if len(prob_ans) > 0: out['predict_last'] = str(prob_ans[-1])
                else: out['predict_last'] = "-10000000000"
                out['is_correct_last'] = 0
                
    for out in question_answer_list:
        if 'is_correct_last' not in out:
            out['is_correct_last'] = out['is_correct']
            out['predict_last'] = out['predict']
    
    return question_answer_list 

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
    
def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

def gather_html(qid):
    with open(f'htmls/GSM8K_html/math_{qid}.json') as f:
        d = json.load(f)
        qid = d['qid']
        soln = d["predict"]
        soln = soln.split('Q:')[0].strip()
        
    return soln