import os
import json
import argparse
import random
import time
from tool import *
from base_prompt_2 import *
from model import *
from utilities import extract_prediction, normalize_answer
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
import pandas as pd
client = OpenAI(api_key = "sk-q8ZyT3BlbkFJrH3V1U4l4VDrkJzgHmSG")

# openai.api_key = os.getenv("OPENAI_API_KEY")


def load_data(args):
    problems_test = json.load(open('../data/tabmwp/finqa_test_formatted_for_promptpg.json'))
    problems_train = json.load(open('../data/tabmwp/finqa_final_correct_train_samples_formatted_for_promptpg.json'))
    problems = {**problems_test, **problems_train}

    # test problem ids
    test_pids = list(problems_test.keys())
    test_pids = test_pids[:args.test_number] if args.test_number > 0 else test_pids
    print(f"number of test problems: {len(test_pids)}\n")

    # pick up shot/in-context example candidates from the training set
    train_pids = list(problems_train.keys())

    #cand_pids = random.sample(train_pids, args.cand_number)  # random sample

    return problems, test_pids, train_pids


def get_gpt3_output(prompt, args):


    print("Check format of prompt")

    print(prompt)

    patience = 100
    while True:
        try: 
            result = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user","content": prompt,}
                            ],
                        model="gpt-4",
                    )
            print(result)
            output = result
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! Running out of patience waiting for OpenAI")
            else:
                print(e)
                time.sleep(0.1)
    return output


def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_seed_{}_finqa_pot_plus_promptpg_gpt4_testing.json".format(result_path, args.label, args.test_split,args.shot_number, args.seed)

    return result_file


def save_results(result_file,cand_pids, args, results):
    data = {}
    # data['acc'] = acc
    # data['correct'] = correct
    # data['count'] = count
    data['cand_pids'] = cand_pids
    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    # parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='test', choices=['dev', 'dev1k', 'test', 'test1k','finqa'])
    parser.add_argument('--test_number', type=int, default=100, help='GPT-3 is expensive. -1 for the whole test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument(
    #     '--prompt_format',
    #     type=str,
    #     default='TQ-A',
    #     choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
    #     help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy Model settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--cand_number', type=int, default=20, help='Number of candidate prompts.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    # problems, test question ids, candidate prompt pids, RL training pids
    problems, pids, cand_pids = load_data(args)



    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the learned check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
    else:
        results = {}

    total = len(pids)
    check_count = len(results)  # number of existing results
    # correct = 0  # number of correct results

    # policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    # print("candidate prompts: ")
    # print("===========")
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems)  # CHECK !!!
        # print(example)
        # print("===========")
        cand_examples.append(example)

    # ======================================================= INFERENCE ===============================================
    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)
        if os.path.exists(ckpt_path):
            # policy_model.linear.load_state_dict(torch.load(ckpt_path))
            policy_model.linear.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        else:
            print(f"The ckpt path for [{ckpt_path}] does not exist!")  # CHECK
            exit()
    else:
        print(f"!!! Load the pre-traind model instead!")  # CHECK
        # exit()

    policy_model.eval()
    output_df = pd.DataFrame(columns = ['Test_Question', 'Test_Answer', 'Test_id', 'Shot_Question', 'Shot_Answer', 'Shot_program', 'Shot_Table'])
    with torch.no_grad():

        # Calculate the embeddings for candidate examples only one time!
        cand_embedding = policy_model(cand_examples)
        # print("cand_embedding:", cand_embedding.shape)  # [cand_num x emb_size]
        correct, wrong = 0, 0
        for i, pid in enumerate(pids):
            count = i + 1  # number of current results
            problem = problems[pid]
            answer = problems[pid]['answer']
            # options = problems[pid]['choices']
            # unit = problems[pid]['unit']

            # print('Problem', problem)
            question = problems[pid]['question']
            id = problems[pid]['id']

            example = create_example_from_pid(pid, problems, test=True)

            ctxt_embedding = policy_model([example])
            # print("ctxt_embedding:", ctxt_embedding.shape)  # [1 x emb_size]

            scores = F.softmax(torch.mm(ctxt_embedding, cand_embedding.t()), dim=1)[0]  # [cand_num]
            # print(scores.shape)
            scores = scores.cpu().detach().numpy().tolist()

            cand_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.shot_number]
            shot_pids = [cand_pids[cid] for cid in cand_ids[::-1]]
            for shot_pid in shot_pids:
                prompt_q = problems[shot_pid]['question']
                prompt_ans = problems[shot_pid]['answer']
                prompt_prog = problems[shot_pid]['program']
                # prompt_id = problems[shot_pid]['id']
                prompt_table = problems[shot_pid]['table']
                rows_to_add = [[question, answer, id, prompt_q, prompt_ans, prompt_prog, prompt_table]]
                new_rows_df = pd.DataFrame(rows_to_add, columns=output_df.columns)
                output_df = pd.concat([output_df, new_rows_df], ignore_index=True)
            
       
        result_path = f"{args.output_root}/{args.model}"
        os.makedirs(result_path, exist_ok=True)

        result_file = "{}/{}_{}_{}_seed_{}_prompts_finqa_random.csv".format(result_path, args.label, args.test_split,args.shot_number, args.seed)
        output_df.to_csv(result_file)
