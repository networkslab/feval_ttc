# FEval-TTC: Fair Evaluation Protocol for Test-Time Compute

This is a repository for **FEval-TTC**, the **F**air **Eval**uation protocol for **T**est-**T**ime **C**ompute.

This evaluation framework features CoT queried for multiple LLMs on a variety of mathematical and reasoning datasets.
The few-shot query process and answer extraction are standardised for every dataset, which eases the burden on researchers in terms of time and money.

## Installation
Please, install this package from the source.
```bash
pip install .
```
It requires `api_responses.zip` (download from [Google Drive](https://drive.google.com/file/d/1bjthQeJhRDxs_M78A5oazCS42iE5spLn/view?usp=sharing)) file containing a database. 
For the following example, let us assume this file is in your code directory.

## Example

```python
from feval_ttc import load, DatasetType, LLMType
    
dataset, [llm1,llm2] = load(DatasetType.SVAMP, [LLMType.LLaMA3B32, LLMType.Qwen72B25], api_path="api_responses.zip")

for question_id, dataentry in dataset:
    print("Question: ", dataentry.question)
    print("True answer: ", dataentry.answer)
    llm1_response = llm1(question_id, N=20)
    print("1st CoT answer: ",  llm1_response.cots[0].answer)
    print("Token cost: ", llm1_response.cots[0].tokens)
    print("USD Cost: ", llm1_response.cots[0].dollar_cost)
```
Refer to `examples` folder for more examples of the benchmark evaluation


# Citing
If you use this protocol in your project, please consider citing:
```
@inproceedings{rumiantsev2025fevalttc,
    title={{FE}val-{TTC}: {Fair Evaluation Protocol for Test-Time Compute}},
    author={Pavel Rumiantsev and Soumyasundar Pal and Yingxue Zhang and Mark Coates},
    booktitle={NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling},
    year={2025},
    url={https://openreview.net/forum?id=Fj9Ge7TdrY}
}
```
