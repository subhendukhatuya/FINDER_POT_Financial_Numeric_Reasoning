import json
from time import sleep
from tqdm import tqdm
import openai
from openai import OpenAI
from tool import finqa_equal, safe_execute
from datetime import datetime
from tool import *
from typing import Dict, Any
import argparse
from collections import Counter
import re

import os
from openai import AzureOpenAI

# Please fill below the endpoint and key
AZURE_OPENAI_ENDPOINT="https://gpt4.openai.azure.com/"
AZURE_OPENAI_KEY="92e69699a263d4e8079dd99326a3c5ba0"

client = AzureOpenAI(
  api_key = AZURE_OPENAI_KEY,
  api_version = "2024-02-01",
  azure_endpoint = AZURE_OPENAI_ENDPOINT
)

def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result.choices):
        text = g.message.content
        print(text)
        to_return.append((text, ''))
    to_return = [r[0] for r in to_return]
    print(to_return)
    return to_return

def extract_last_variable(ans):
    if re.search(r'[\[\(].*[\]\)]', ans):
        ans = re.sub(r'^ans\s*=\s*[\[\(]|[)\]]$', '', ans)
        parts = [part.strip() for part in ans.split(',')]
        return parts[-1]
    else:
        # For simple variable assignments, just return 'ans'
        return 'ans'

def extract_last_question(question):
    # Split the string by common punctuation marks used for questions
    parts = re.split(r'[?]', question)
    
    # Remove any empty strings and strip whitespace
    parts = [part.strip() for part in parts if part.strip()]
    
    # Return the last question
    return parts[-1] if parts else ""

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()

# def create_reader_request_processed(example: Dict[str, Any]):
#     prompt = 'Read the following table and probable relevant facts, and then write code to answer a question, the answer can be a float/int or bool:\n'
#     prompt += 'Table: ' + '\n' + example['table'] + '\n'
#     prompt += "Probable relevant facts:\n "
#     for element in example['retrieved']:
#         prompt+= element + '\n'
#     prompt += '\n'  
#     prompt += 'Question: {}\n'.format(example['question'])
#     prompt += '#Python code below\n'
#     print("method called")
#     return prompt

def build_prompt(row: Dict[str, Any]):
    prompt = 'Read the following table and probable relevant facts, and then write the code to answer the final question. The answer can be a float/int or bool:\n'
    prompt += 'Table: ' + '\n' + example['table'] + '\n'
    prompt += "Probable relevant facts:\n "
    for element in example['retrieved']:
        prompt+=element+'\n'
    prompt += 'All Question(s): {}\n'.format(row['question'])
    prompt += 'Final Question: {}\n'.format(extract_last_question(row['question']))
    prompt += '#Python code below (Give answer to the final question only and no comments or explanation):\n'
    return prompt

if __name__ == "__main__":
    with open('Data_Target_Module/ConvFinqa_turn/output/best_matched_with_retrieved_facts_and_questions_flan_convfinqa_3shot.json') as f:
        finqa_dev = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    if args.greedy:
        filename = f'Experiment/Final/Convfinqa_greedy_flan_default_3shot_2.jsonl'
    else:
        filename = f'Experiment/Final/finqa_self_consistency.jsonl'
    count = 1
    for example in tqdm(finqa_dev):
        writer = open(filename, 'a+')
        print('\nsample: '+ str(count)+ '\n')
        count +=1
        full_prompt = example['prompt'] + "\n\n"
        full_prompt += build_prompt(example)
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            break

        if args.greedy:
            # greedy decoding
            got_result = False
            if not got_result:
                try:
                    result = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user","content": full_prompt,}
                            ],
                        model="gpt4",
                    )
                    print(result, flush=True)
                    got_result = True
                    sleep(2)
                except Exception:
                    print(e)
                    print('exception thrown')
                    writer.close()
                    sleep(6)
        else:
            # self-consistency decoding
            got_result = False
            print("entered here")
            while not got_result:
                try:
                    result = client.completions.create(engine='gpt-4',
                    prompt=full_prompt,
                    max_tokens=512,
                    temperature=0.5,
                    top_p=1,
                    n=30,
                    stop=['\n\n'],
                    logprobs=1)
                    got_result = True
                except Exception as e:
                    print("exception thrown")
                    sleep(3)

        # self-consistency decoding or greedy decoding.
        print("verifying")
        print(result)
        result_counter = Counter()
        codes = parse_api_result(result)
        print(codes)
        codes = [code.replace('&', '_') for code in codes]
        for r in codes:
            r = r.replace('```python\n', '')
            r = r.replace('\n```', '')
            lines = r.split("\n")
            last_line = lines[-1]
            if(len(lines) > 1):
                if(last_line == ''):
                    last_line = lines[-2]
            key = extract_last_variable(last_line)
            ans = safe_execute(r, key)
            ans = floatify_ans(ans)
            if ("* 100 " in r or "* 100\n" in r or "*100 " in r or "*100\n" in r):
                qn = extract_last_question(example['question'])
                if("percent" in qn or "percentage" in qn or "growth rate" in qn):                        
                    try:
                        ans = ans/100
                    except:
                        pass
            if ans is not None:
                result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = None

        # Further Process according to FinQA dataset
        if prediction is not None:
            if type(prediction) == bool:
                if prediction:
                    prediction = 'yes'
                else:
                    prediction = 'no'
            elif type(prediction) == list:
                prediction = prediction[0]
            else:
                assert type(prediction) in [float, int, str], prediction

        if prediction is None:
            wrong += 1
        elif finqa_equal(prediction, example['answer'], False):
            correct += 1
        else:
            wrong += 1
        if(count <5 ):
            print('sample answer')
            print(codes)
            print(prediction)
        

            

        print('accuracy: ', correct / (correct + wrong))
        acc = correct / (correct + wrong)
        example.update({'generated': codes, 'executed': prediction,'accuracy': acc})
        writer.write(json.dumps(example) + '\n')
        writer.close()
        

    print()
    print('accuracy: ', correct / (correct + wrong))
    writer.close()

