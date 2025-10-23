from collections import Counter
from tqdm import tqdm

from feval_ttc import load, DatasetType, LLMType, LLMResponse


def get_self_consistency_answer(llm_response: LLMResponse) -> str | None:
    filtered_answers = [a for a in llm_response.answers if a is not None]
    if len(filtered_answers) == 0:
        return None
    return Counter(filtered_answers).most_common(1)[0][0]


dataset, [llm, ] = load(DatasetType.AQUA, [LLMType.Mixtral8x7B], api_path="api_responses.zip")


accuracy = 0.0
dollar_cost = 0.0
for question_id, dataentry in tqdm(dataset):
    response = llm(question_id, N=20)
    llm_answer = get_self_consistency_answer(response)
    true_answer = dataentry.answer
    
    is_correct = true_answer == llm_answer
    # print(f"True: {true_answer} | Model: {llm_answer}")
    accuracy += float(is_correct)
    dollar_cost += response.dollar_cost

accuracy /= len(dataset)
    
print(f"Accuracy SC: {accuracy:.3f}")
print(f"Dollar cost SC: ${dollar_cost:.9f}")
