import os 
import shutil
import re
import json

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
    
def return_predicted_answer(question_answer_list):
    for out in question_answer_list:
        soln = out['response'].split('Q:')[0]
        exact = float(out['answer'])
        
        if 'The answer is' in soln:
            soln = soln.split('The answer is')[-1]
            prob_ans = re.findall(r"[-+]?(?:[0-9,]*\.\d+)", soln)
            prob_ans = [float(x.replace(',', '')) for x in prob_ans]
            prob_ans = [float(x) for x in prob_ans]
            if len(prob_ans) > 0 and exact == prob_ans[0]:
                out['predict'] = out['answer']
                out['is_correct'] = 1
            else:
                if len(prob_ans) > 0: out['predict'] = str(prob_ans[0])
                else: out['predict'] = "-10000000000"
                out['is_correct'] = 0
        else:
            out['predict'] = "-10000000000"
            out['is_correct'] = 0


        soln = out['response'].split('Q:')[0]
        exact = float(out['answer'])
        prob_ans = re.findall(r"[-+]?(?:[0-9,]*\.\d+)", soln)
        prob_ans = [float(x.replace(',', '')) for x in prob_ans]
        if len(prob_ans) > 0 and exact == prob_ans[-1]:
            out['predict_last'] = out['answer']
            out['is_correct_last'] = 1
        else:
            if len(prob_ans) > 0: out['predict_last'] = str(prob_ans[-1])
            else: out['predict_last'] = "-10000000000"
            out['is_correct_last'] = 0
            
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