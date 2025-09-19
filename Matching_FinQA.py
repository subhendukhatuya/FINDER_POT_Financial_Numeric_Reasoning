import json
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


with open("Data_Target_Module/Finqa/finqa_test_with_table_text.json") as json_file :
    data = json.load(json_file)
# file_path = 'Data/lora_flan_retriever/lora_flan_large_corrected_prediction_finqa_rel_fact.txt'
file_path = 'Data_Target_Module/mistral_retriever/mistral_finqa_rel_fact_file.txt'
with open(file_path,'r+') as f:
    content = f.readlines()
content_list = []
print(type(data[275]['table_text']) == float)
for i in range(len(content)):
    index =content[i].find('Pred:')
    s= content[i][index+5:].split(';')
    s= null_remover(s)
    data_dict = data[i]
    text = sent_tokenize(data_dict['text'])
    table=[]
    if type(data_dict['table_text']) != float:
        table = data_dict['table_text'].split(';')
    context_sentences = text
    context_sentences += (table)
    context_sentences = null_remover(context_sentences)
    target = model.encode(context_sentences)
    
    sentences = []
    for sent in s:
        matched = matching_sent(sent, context_sentences, target)
        sentences.append(matched)
    matched_question= []
    matched_question = top_three(data_dict['question'], context_sentences, target)
    data_dict['retrieved']= list(set(sentences + matched_question))
    content_list.append(data_dict)
with open('Data_Target_Module/mistral_retriever/output/best_matched_with_retrieved_facts_and_questions_mistral.json','w') as file:
    json.dump(content_list, file)
        
