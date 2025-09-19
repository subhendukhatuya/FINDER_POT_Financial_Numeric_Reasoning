import os
os.environ["HF_TOKEN"] = "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"
import pandas as pd

import torch
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
from torch.utils.data import Dataset, DataLoader

class TourismDataset(Dataset):
    """Tourism Dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.

        """
        self.tacos_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sentences = self.tacos_df['input']
        self.text_labels = self.tacos_df['Rel_Fact']

    def __len__(self):
        return len(self.tacos_df)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        text_label = self.text_labels[idx]

        sample = {'sentence': sentence,  'text_label': text_label}

        return sample

class TestDataset(Dataset):
    """Tourism Dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.

        """
        self.tacos_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sentences = self.tacos_df['input']
        self.text_labels = self.tacos_df['Rel_Fact']


    def __len__(self):
        return len(self.tacos_df)



    def __getitem__(self, idx):

        sentence = self.sentences[idx]
        text_label = self.text_labels[idx]

        sample = {'sentence': sentence,  'text_label': text_label}

        return sample

tacos_dataset_train = TourismDataset(csv_file='../Data_Retriever_Module/convfinqa_train_rel_fact_instruction.csv',
                                     root_dir='./')

tacos_dataset_test = TestDataset(csv_file='../Data_Retriever_Module/convfinqa_test_rel_fact_instruction.csv', root_dir='./')



from datasets import Dataset

dataset_train = Dataset.from_dict(
        {"sentence": list(tacos_dataset_train.sentences)[0:5000], "text_label": list(tacos_dataset_train.text_labels)[0:5000]})
dataset_test = Dataset.from_dict(
        {"sentence": list(tacos_dataset_test.sentences), "text_label": list(tacos_dataset_test.text_labels)})

print(len(dataset_train),flush=True)


def transform_conversation(example):
    instruction = example['sentence']
    output = example['text_label']

    final_instruction = instruction + "##Label Descriptions: " + output


    return {'instruction': final_instruction}


# Apply the transformation
transformed_dataset = dataset_train.map(transform_conversation)

# Model
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
#Fine-tune model name
new_model = "mistral-convfinqa"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

#Configration of QLoRA
#Quantization Configuration
#To reduce the VRAM usage we will load the model in 4 bit precision and we will do quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #Quant type
    #We will use the "nf4" format this was introduced in the QLoRA paper
    bnb_4bit_quant_type="nf4",
    #As the model weights are stored using 4 bits and when we want to compute its only going to use 16 bits so we have more accuracy
    bnb_4bit_compute_dtype=torch.float16,
    #Quantization parameters are quantized
    bnb_4bit_use_double_quant=True,
)


# LoRA configuration
peft_config = LoraConfig(
    #Alpha is the strength of the adapters. In LoRA, instead of training all the weights, we will add some adapters in some layers and we will only
    #train the added weights
    #We can merge these adapters in some layers in a very weak way using very low value of alpha (using very little weight) or using a high value of alpha
    #(using a big weight)
    lora_alpha=15,
    #10% dropout
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    cache_dir = '/NS/ssdecl/work',
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
#prepare_model_for_kbit_training---> This function basically helps to built the best model possible
model = prepare_model_for_kbit_training(model)

# Set training arguments
training_arguments = TrainingArguments(
        output_dir="./results_convfinqa",
        num_train_epochs=4,
        per_device_train_batch_size=4,# Number of batches that we are going to take for every step
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=625,
        logging_steps=25,
        optim="paged_adamw_8bit",#Adam Optimizer we will be using but a version that is paged and in 8 bits, so it will lose less memory
        learning_rate=5e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        max_steps=-1,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=transformed_dataset,
    eval_dataset=transformed_dataset,
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
new_model_2 = "mistral-convfinqa"
trainer.model.save_pretrained(new_model_2)

print('training completed')

