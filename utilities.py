import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder = '/NS/ssdecl/work')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def matching_sent(input_sent,target_sentences, embedded_target_sentences):
    embedded_input = model.encode(input_sent)
    #sentences = ['hello.',' What is your name? ','What is my name?','What is her name?']
    #all_sent_embedding = model.encode(target_sentences)
    cosine_similarities_pred_all = util.dot_score(embedded_input, embedded_target_sentences)
    values, indices = torch.topk(cosine_similarities_pred_all, 1)
    values = values.tolist()
    indices= indices.tolist()
    print(values)
    print(indices)
    return target_sentences[indices[0][0]]

def null_remover(slist):
    nlist=[]
    for s in slist:
        if len(s)>1:
            nlist.append(s)
    return nlist

def top_three(input_sent,target_sentences, embedded_target_sentences):
    embedded_input = model.encode(input_sent)
    #sentences = ['hello.',' What is your name? ','What is my name?','What is her name?']
    #all_sent_embedding = model.encode(target_sentences)
    cosine_similarities_pred_all = util.dot_score(embedded_input, embedded_target_sentences)
    values, indices = torch.topk(cosine_similarities_pred_all, 3)
    values = values.tolist()
    indices= indices.tolist()
    print(values)
    print(indices)
    matched = []
    for i in range(3):
        matched.append(target_sentences[indices[0][i]])
    return matched
