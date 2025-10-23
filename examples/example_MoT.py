from collections import Counter
from tqdm import tqdm
from typing import Tuple

from feval_ttc import load, DatasetType, LLMType, LLMResponse



SAMPLE_BUDGET = 20
THRESHOLD = 0.9


def get_MoT_1D_Vote(weak_llm_response: LLMResponse,
                    strong_llm_response: LLMResponse,
                    threshold: float) -> Tuple[str, float]:
    weal_llm_answer, weal_llm_confidence = Counter(weak_llm_response.answers).most_common(1)[0]
    weal_llm_confidence /= len(weak_llm_response)
    if weal_llm_confidence > threshold and weal_llm_answer is not None:
        return weal_llm_answer, weak_llm_response.dollar_cost
    
    strong_llm_answer = Counter(strong_llm_response.answers).most_common(1)[0][0]
    return strong_llm_answer, strong_llm_response.dollar_cost + weak_llm_response.dollar_cost



dataset, [weak_llm, strong_llm] = load(DatasetType.BIGBENCH_DATE, [LLMType.Qwen1B25, LLMType.Qwen72B25], api_path="api_responses.zip")

accuracy = 0.0
llm_dollar_cost = 0.0
for question_id, dataentry in tqdm(dataset):
    weak_llm_response = weak_llm(question_id, N=SAMPLE_BUDGET)
    strong_llm_response = strong_llm(question_id, N=SAMPLE_BUDGET)
    llm_answer, llm_cost = get_MoT_1D_Vote(weak_llm_response, strong_llm_response, THRESHOLD)
    true_answer = dataentry.answer
    
    is_correct = true_answer == llm_answer
    # print(f"True: {true_answer} | Model: {llm_answer}")
    accuracy += float(is_correct)
    llm_dollar_cost += llm_cost

accuracy /= len(dataset)
    
print(f"Accuracy MoT: {accuracy:.3f}")
print(f"Dollar cost MoT: ${llm_dollar_cost:.9f}")

