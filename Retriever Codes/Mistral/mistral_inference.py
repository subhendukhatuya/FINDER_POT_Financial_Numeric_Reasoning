# Reload model in FP16 and merge it with LoRA weights

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["HF_TOKEN"] = "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"
model_name =  "mistralai/Mistral-7B-Instruct-v0.2"
#Fine-tune model name
new_model ="mistral-convfinqa"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir='/NS/ssdecl/work',
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir='/NS/ssdecl/work')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"







tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=2048,
    padding_side="left",
    add_eos_token=True,cache_dir='/NS/ssdecl/work')

tokenizer.pad_token = tokenizer.eos_token


#df_2 = pd.read_csv('test_tacos_t2.csv')
df = pd.read_csv('../Data_Retriever_Module/convfinqa_train_rel_fact_instruction.csv')
predicted_tuples = []
for i, row in df.iterrows():
    if(i% 1000 == 0):
        print(i, flush = True)
    #review = row['Sentence']
    review = row['input'] + "##Label Descriptions:"
    #gt = row['Target']
    model_input = tokenizer(review, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        predicted_text = tokenizer.decode(model.generate(**model_input,  max_new_tokens=150)[0], skip_special_tokens=True)
    predicted_tuples.append(predicted_text)

df ['Predicted'] = predicted_tuples

df.to_csv('train_result_convfinqa_mistral_5e4.csv')

