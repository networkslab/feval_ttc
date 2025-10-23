from feval_ttc import ModelCost, LLMType


INPUT_TEXT = "How to find files by mask from a python script?"
OUTPUT_TEXT = """To find files matching a specific pattern (or "mask") in Python, you can use several standard modules: `glob` or `pathlib`. 

* Use `glob.glob('*.txt')` for non-recursive searches.
* Use `glob.glob('**/*.txt', recursive=True)` or `pathlib.Path().rglob('*.txt')` for recursive searches.
"""

model_cost = ModelCost(LLMType.GPT4oMINI)
input_cost = model_cost.input_dollar_cost(INPUT_TEXT)
output_cost = model_cost.output_dollar_cost(OUTPUT_TEXT)
print(f"User input cost: ${input_cost:.09f}")
print(f"LLM response cost: ${output_cost:.09f}")
