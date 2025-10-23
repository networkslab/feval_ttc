from feval_ttc import available_models, DatasetType


llms = available_models(DatasetType.MATH_500, api_path="api_responses.zip")
print(f"Dataset: ", DatasetType.MATH_500)
for m in llms:
    print(m)

