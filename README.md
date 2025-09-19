# Program of Thoughts for Financial Reasoning
This repository is the code base for the EMNLP 2025 paper Program of Thoughts for Financial Reasoning: Leveraging Dynamic In-Context Examples and Generative Retrieval. In the paper we propose FINDER, a novel two-step framework designed to improve large language models’ (LLMs) capabilities in financial numerical reasoning.
- A generative retriever extracts relevant facts from unstructured data sources, including both text and tables. (Retriever module)
- A context-aware Program of Thought prompting mechanism dynamically selects in-context examples to guide reasoning. (Dynamic Context Selection)

Using this approach, FINDER achieves state-of-the-art performance on the FinQA and ConvFinQA datasets, with execution accuracy improvements of +5.98% and +4.05%, respectively, over previous benchmarks.

## Running the Code
### Data Folder

**DATA Folder:** https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link

Unzip "Data_Target_Module" folder in the root folder


### Requirements

Install all requirements listed in requirements.txt


## Retriever Module

These codes are present under the folder "Retriever Codes". The outputs of these codes will be used in the Target Computation module. \\ 
The Target Computation Module can be directly run as we provide the outputs from the Retrievers as "_Data_Target_Module_" folder \\

For running Retriever Codes please refer to Readme in the "Retriever Codes" folder

### Run the Matching codes:

For FinQA: 
```
python3 Matching_FinQA.py
```

For ConvFinQA: 
```
python3 Matching_ConvFinQA.py

```
## In Context Example selection Module

These codes are present under the folder "In_Context_Selection".
For running In Context Example selection Codes please refer to Readme in the "In_Context_Selection" folder

### Running Target Answer Computation Module:

Before running these codes please set up GPT-4 API Key and update that in the code.

For FinQA: 
```
python3 finqa_run.py
```

For ConvFinQA: 
```
python3 convfinqa_run.py
```

## Output

We report all the final files generated under Experiment/Final.

To get final accuracy and metrics run final_execution_checker.py
