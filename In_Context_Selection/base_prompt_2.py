def get_table_text(problem):
    table = 'Read the following text and table, and then write Python code to answer a question, the answer can be a float/int or bool:\n'
    table += problem['table']
    # title = problem['table_title']
    # if title and len(title) > 0:
    #     table = f"[TITLE]: {title}\n{table}"
    return table


def get_question_text(problem):
    question = problem['question']

    # unit = problem['unit']
    # if unit and len(unit) > 0:
    #     question = f"{question} (Unit: {unit})"

    # choices = problem['choices']
    # if choices and len(choices) > 0:
    #     choice_list = []
    #     for i, c in enumerate(choices):
    #         choice_list.append("({}) {}".format(option_inds[i], c))
    #     options = " ".join(choice_list)
    #     #print(options)
    #     question = f"{question}\nOptions: {options}"

    return question


def get_answer(problem):
    return problem['answer']
    
def get_relfact(problem, test = False):
    facts = ""
    if(test == False):
        facts = problem['text']
    else:
        for line in problem['retrieved']:
            facts += line
            facts += '\n'
    return facts


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['program']
    return solution


def create_one_example( table, question, answer, rel_fact, solution, test_example=True):

    # input_format, output_format = format.split("-")  # e.g., "TQ-A"

    # elements = {
    #     "Q": f"Question: {question}",
    #     "T": f"Table: {table}",
    #     "S": f"Solution: {solution}",
    #     "A": f"Answer: The answer is {answer}.",
    #     "AS": f"Answer: The answer is {answer}. BECAUSE: {solution}",
    #     "SA": f"Answer: {solution} The answer is {answer}."
    # }

    # Input
    input = table + "\nProbable relevant facts:\n" + rel_fact + '\nQuestion: '+ question

    # Output
    if test_example:
        output = "#Python code below:"
    else:
        output = "#Python code below:\n" + solution

    # Prompt text
    text = input + "\n" + output
    text = text.replace("  ", " ").strip()

    return text


def build_prompt(problems, shot_pids, test_pid, args):

    examples = []
    pids = shot_pids + [test_pid]

    # n-shot training examples
    for pid in pids:
        problem = problems[pid]
        table = get_table_text(problem)
        question = get_question_text(problem)
        answer = get_answer(problem)
        solution = get_solution_text(problems[pid])

        if pid == test_pid:
            assert pid not in shot_pids
            rel_fact = get_relfact(problem, test = True)
            example = create_one_example( table, question, answer, rel_fact, solution, test_example=True)
        else:
            rel_fact = get_relfact(problem, test = False)
            example = create_one_example( table, question, answer, rel_fact, solution, test_example=False)

        examples.append(example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def create_example_from_pid(pid, problems, test=False):
    problem = problems[pid]
    table = get_table_text(problem)
    question = get_question_text(problem)
    answer = get_answer(problem)

    if test:
        rel_fact = get_relfact(problem, test = True)
        solution = ""
        example = create_one_example( table, question, answer, rel_fact, solution, test_example=True)
    else:
        solution = get_solution_text(problems[pid])
        rel_fact = get_relfact(problem, test = False)
        example = create_one_example( table, question, answer, rel_fact, solution, test_example=False)

    return example

def parse_api_result(result):
    to_return = []
    text_content = result.choices[0].message.content
    for idx, g in enumerate(result.choices):
        text = g.message.content
        # logprob = sum(g.logprobs.token_logprobs)
        to_return.append((text, ""))
    to_return = sorted(to_return, key=lambda tup: tup[1], reverse=True)
    to_return = [r[0] for r in to_return]
    return to_return,text_content
