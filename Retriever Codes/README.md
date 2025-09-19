Please follow the below steps to run our codebase.

## Retriever Module
This module retrieves the relevant facts corresponding to the question. We utilize FLAN-T5 backbone to retrieve facts.
### Data
Data Folder: https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link
First unzip the folder "Data_Retriever" in this folder. Also unzip the folder "Data_Ensemble" under Ensemble folder. 

### For FLAN-T5 (All codes present under FLAN-T5 Folder):

For FinQA:
```
python3 lora_flan_large_finqa_rel_fact.py
```
For ConvFinQA: 
```
python3 lora_flan_large_convfinqa_rel_fact_train.py
```

### For Mistral (All codes present under Mistral Folder):

For training: 
```
python3 mistral_train.py
```
For inference: 
```
python3 mistral_inference.py
```



