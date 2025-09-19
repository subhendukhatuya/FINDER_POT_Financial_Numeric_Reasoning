import json
import nltk
#nltk.download('punkt')
from nltk import sent_tokenize
import numpy as np
import pandas
import pandas as pd
import torch
from utilities import *
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder = '/NS/ssdecl/work')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ast

content_list = []
df = pd.read_csv("Data_Target_Module/ConvFinqa_turn/convfinqa_test_file.csv")
df['Table'] = df['Table'].apply(ast.literal_eval)
df['Pre_Text'] = df['Pre_text'].apply(ast.literal_eval)
df['Post_Text'] = df['Post_text'].apply(ast.literal_eval)

file_path = "Data_Target_Module/ConvFinqa_turn/Convfinqa_ensemble_rel_fact_file.txt"
with open(file_path,'r+') as f:
    content = f.readlines()
for index, row in df.iterrows():
    data_dict = {}
    ind =content[index].find('Pred:')
    s= content[index][ind+5:].split(';')
    s= null_remover(s)
    data_dict['text'] = ""
    for line in row['Pre_Text']:
        if len(line) > 0:
            data_dict['text'] += ' '+ line
    for line in row['Post_Text']:
        if len(line) > 0:
            data_dict['text'] += ' '+ line
    text = data_dict['text']
    data_dict['table_text'] = row['Table_Text']
    text = sent_tokenize(text)
    table=[]
    if type(row['Table_Text']) != float:
        table = row['Table_Text'].split(';')
    context_sentences = text
    context_sentences += (table)
    context_sentences = null_remover(context_sentences)
    target = model.encode(context_sentences)
    data_dict['table'] = '' 
    for t in row['Table']:
        line = ""
        # print(t)
        for table_line in t:
            # print(table_line)
            line += table_line + '|'
        line = line.strip()
        # print(line)
        line = line[:len(line)-1]
        # print(line)
        data_dict['table'] += '\n'+ line
    sentences = []
    # counterrrr=0
    for sent in s:
        # counterrrr +=1
        # print(s)
        matched = matching_sent(sent, context_sentences, target)
        sentences.append(matched)
    data_dict['question'] = row['Question']
    try:
        data_dict['answer'] = float(row['Answer'])
    except:
        data_dict['answer'] = row['Answer']
    # data_dict['answer'] = row['GT_Answer']
    matched_question= []
    matched_question = top_three(data_dict['question'], context_sentences, target)
    data_dict['retrieved']= list(set(sentences + matched_question))
    content_list.append(data_dict)

with open('Data_Target_Module/ConvFinqa_turn/output/best_matched_with_retrieved_facts_and_questions_ensemble_bart.json','w') as file:
    json.dump(content_list, file)

