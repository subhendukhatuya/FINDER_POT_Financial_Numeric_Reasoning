Please follow the below steps to run our codebase.

## Dynamic Context Selection

This module dynamically selects candidate examples that can best guide the GPT engine to generate the best answer. We integrate Program of thought based prompting into the existing PromptPG [[1]](#1) framework.

### Data
Unzip the folder "Data_Prompt_Dynamic" from https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link

### Code to train policy:

Before running these codes please set up Azure endpoints for GPT-4 and paste the API Key in the codes below.

```
pip install -r requirements.txt
```
```
cd run_gpt3_rl
```
```
python learn_policy.py --label exp2 --ckpt_root ../checkpoints --shot_number 2 --prompt_format TQ-SA --seed 2 --model_config bert-base-uncased --train_number 40 --cand_number 10 --lr 0.001 --epochs 20 --embedding_size 128 --batch_size 20 --gpu 0
```

### Inference

```
python3 examples_extraction.py \--label exp4 \--ckpt_root ../checkpoints \--model gpt3_rl \--test_split test \--test_number -1 \--shot_number 4 \--seed 2 \--cand_number 50 \--embedding_size 128 \--model_config bert-base-uncased \--ckpt exp4/ckpt_best_reward_2.pt \--gpu 0
```

## References

<a id="1">[1]</a>
```bibtex
@inproceedings{lu2023dynamic,
  title={Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning},
  author={Lu, Pan and Qiu, Liang and Chang, Kai-Wei and Wu, Ying Nian and Zhu, Song-Chun and Rajpurohit, Tanmay and Clark, Peter and Kalyan, Ashwin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```




