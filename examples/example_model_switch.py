import itertools
from collections import Counter
from tqdm import tqdm
from typing import Tuple, List
from math import log

from feval_ttc import load, DatasetType, LLMType, LLMResponse


SAMPLE_BUDGET = 5


def calculate_weight(answers):
    l = len(answers)

    unique_weights = [a[1] for a in Counter(answers).most_common()]
    norm = sum(unique_weights)
    entropy = - sum( w * log(w) / norm   for w in unique_weights)
    weight = 1 / l + (1 - 1/l)*(1 - entropy / log(l))
    return [weight] * l


def get_ModelSwitch_answer(llm_responses: List[LLMResponse]) -> Tuple[str, float]:
    dollar_cost = 0.0
    all_answers = []
    all_answer_w = []
    for i, response in enumerate(llm_responses):
        dollar_cost += response.dollar_cost
        fileterd_answers = [a for a in response.answers if a is not None]
        counts_ = Counter(fileterd_answers)
        answer = counts_.most_common(1)[0][0]
        if len(counts_) == 1 and i < len(llm_responses)-1:
            return answer, dollar_cost
        all_answers.extend(fileterd_answers)
        all_answer_w.extend(calculate_weight(fileterd_answers))
        
    weighted_results = []
    for a, ws in itertools.groupby(zip(all_answers, all_answer_w), key=lambda x: x[0]):
        weighted_results.append((a, sum(w[1] for w in ws)))
    weighted_results.sort(key=lambda x: x[1])  
    
    return weighted_results[-1][0], dollar_cost


dataset, llms = load(DatasetType.BIGBENCH_RUIN_NAMES, [LLMType.Qwen1B25, LLMType.LLaMA70B33, LLMType.GPT4oMINI], api_path="api_responses.zip")

accuracy = 0.0
llm_dollar_cost = 0.0
for question_id, dataentry in tqdm(dataset):
    llm_responses = [llm(question_id, N=SAMPLE_BUDGET) for llm in llms]
    llm_answer, llm_cost = get_ModelSwitch_answer(llm_responses)
    true_answer = dataentry.answer
    
    is_correct = true_answer == llm_answer
    # print(f"True: {true_answer} | Model: {llm_answer}")
    accuracy += float(is_correct)
    llm_dollar_cost += llm_cost

accuracy /= len(dataset)
    
print(f"Accuracy ModelSwitch: {accuracy:.3f}")
print(f"Dollar cost ModelSwitch: ${llm_dollar_cost:.9f}")

