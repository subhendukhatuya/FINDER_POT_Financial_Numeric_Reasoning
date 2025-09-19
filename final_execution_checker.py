import json
from time import sleep
from tqdm import tqdm
import openai
from tool import finqa_equal, safe_execute, floatify_ans, get_precision
from datetime import datetime
from tool import *
from typing import Dict, Any
import argparse
from collections import Counter
import os
import pandas as pd
import re

wrong = 0
correct = 0
count = 0

with open("Experiment/Final/Finqa/finqa_flan_promptpg_default.jsonl",'r') as json_file:
    json_list = list(json_file)
list_data=[]

for json_str in json_list:
    result = json.loads(json_str)
    list_data.append(result)

for ind, example in enumerate(list_data):
    prediction = example['executed']
    if prediction is None:
        wrong += 1

    elif finqa_equal(prediction, example['answer'], False):
        correct += 1
    else:
        wrong += 1
acc = correct / (correct + wrong)
print("Final Accuracy: ")
print(acc)
print(correct)
print(wrong)