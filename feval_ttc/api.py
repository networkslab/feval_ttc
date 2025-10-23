import os
import zipfile
from .dataset_interface import Dataset, DatasetType
from .llm_interface import LLM, LLMType
from typing import List, Tuple
from pydantic import BaseModel


class LLMTypeList(BaseModel):
    llms: List[LLMType]
    

def load(datatype: DatasetType, llms: List[LLMType], api_path) -> Tuple[Dataset, List[LLM]]:
    '''Loads a dataset and llm records corresponding to this dataset.
    It requires 'api_responses.zip' file containing a database. 
    The location of the file can be customly provided.
    '''
    if not api_path.endswith(".zip"):
        raise ValueError(f"{api_path} is not a valid zip archive")
    
    if not os.path.isfile(api_path):
        raise ValueError(f"{api_path} does not exist")
    
    models = available_models(datatype, api_path=api_path)
    for llmname in llms:
        if llmname not in models:
            raise ValueError(f"Model {llmname} is not available for {datatype}")
    
    with zipfile.ZipFile(api_path, mode="r") as zip:
        with zip.open(f"dataset_{datatype}.txt", mode="r") as file:
            dataset = Dataset.model_validate_json(file.read().decode('utf-8'))
        llm_list = []
        for llmname in llms:
            with zip.open(f"LLM_{llmname}_dataset_{datatype}.txt", mode="r") as file:
                llm_list.append(LLM.model_validate_json(file.read().decode('utf-8')))
        return dataset, llm_list


def available_models(datatype: DatasetType, api_path) ->  List[LLMType]:
    '''Returns a list of models that are available to query for a given dataset.
    '''
    if not api_path.endswith(".zip"):
        raise ValueError(f"{api_path} is not a valid zip archive")
    
    if not os.path.isfile(api_path):
        raise ValueError(f"{api_path} does not exist")
        
    with zipfile.ZipFile(api_path, mode="r") as zip:
        with zip.open(f"dataset_{datatype}_models.txt", mode="r") as file:
            dataset = LLMTypeList.model_validate_json(file.read().decode('utf-8'))
            
    return dataset.llms
