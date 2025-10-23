import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from feval_ttc import load, DatasetType, LLMType


GOOD_TOKEN = '+'
BAD_TOKEN = '-'
STEP_TAG = 'ки'
os.environ["C_INCLUDE_PATH"] = "/usr/local/cuda/include"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def step_tag_inserter(text: str, step_tag: str) -> str:
    sentences = re.split(r'(?<=[\.!?])(?:[ \t]+|(?=[A-Z]\n))', text.strip())
    if sentences[-1] != "":
        sentences.append("")
    return f" {step_tag}".join(sentences)

def run_best_of_N(question, cots):
    scores_answers = []
    for cot in cots:
        if cot.answer is not None:
            inputs_for_prm = f"{question} {step_tag_inserter(cot.raw_text, STEP_TAG)}"
            input_ids = tokenizer(inputs_for_prm, return_tensors='pt', padding=True)
            input_ids, attention_mask = input_ids["input_ids"].cuda(), input_ids["attention_mask"].cuda()
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_scores = scores[0][input_ids[0] == step_tag_id].tolist()
                scores_answers.append((step_scores[-1], cot.answer))
    return max(scores_answers, key=lambda x: x[0])[1]


tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"
candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:] # [648, 387]
step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1] # 12902

model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm', torch_dtype=torch.bfloat16, device_map="auto").eval()

dataset, [llm, ] = load(DatasetType.AQUA, [LLMType.Mixtral8x22B], api_path="api_responses.zip")

accuracy = 0.0
dollar_cost = 0.0
for question_id, dataentry in tqdm(dataset):
    response = llm(question_id, N=20)
    llm_answer = run_best_of_N(dataset[question_id].question, response.cots)
    
    true_answer = dataentry.answer
    is_correct = true_answer == llm_answer
    # print(f"True: {true_answer} | Model: {llm_answer}")
    accuracy += float(is_correct)
    dollar_cost += response.dollar_cost

accuracy /= len(dataset)
    
print(f"Accuracy BestOfN: {accuracy:.3f}")
print(f"Dollar cost BestOfN: ${dollar_cost:.9f}")