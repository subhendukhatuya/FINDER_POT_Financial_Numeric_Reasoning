def get_table_text(problem):
    table = 'Read the following table and probable relevant facts, and then the python code below that answers the final question in the questions. The answer can be a float/int or bool:\n'
    table += problem['table']
    return table


def get_question_text(problem):
    question = problem['question']
    return question


def get_answer(problem):
    return problem['answer']
    
def get_relfact(problem, test = False):
    facts = ""
    if(test == False):
        facts = problem['rel_fact']
    else:
        for line in problem['retrieved']:
            facts += line
            facts += '\n'
    return facts


def get_solution_text(problem):
    solution = problem['program']
    return solution


def create_one_example( table, question, answer, rel_fact, solution, test_example=True):
    # Input
    input = table + "\nProbable relevant facts:\n" + rel_fact + '\nQuestion(s): '+ question

    # Output
    if test_example:
        output = "#Python code below (Give answer to the final Question only):"
    else:
        output = "#Python code below (Give answer to the final Question only)\n" + solution

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
