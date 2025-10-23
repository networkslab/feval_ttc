from feval_ttc import load, DatasetType, LLMType


dataset, [llm, ] = load(DatasetType.CommonsenseQA, [LLMType.Mixtral8x7B], api_path="api_responses.zip")

question_id = 42
response = llm(question_id, N=20)

print(f"Request processing dollar cost: ${response.request.dollar_cost:0.9f}")
print(f"Request processing token cost: {response.request.token_cost}")
print(f"First CoT response dollar cost: ${response.cots[0].metadata.dollar_cost:0.9f}")
print(f"First CoT response token cost: {response.cots[0].metadata.token_cost}")
print(f"Total LLM query cost: ${response.dollar_cost:0.9f}")
