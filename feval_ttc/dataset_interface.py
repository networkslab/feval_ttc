from enum import StrEnum
from pydantic import BaseModel
from typing import List, Optional


class DatasetType(StrEnum):
    AQUA = "AQuA"
    CommonsenseQA = "CommonsenseQA"
    GSM8K = "GSM8K"
    SVAMP = "SVAMP"
    
    BIGBENCH_DATE = "Bigbench_Date"
    BIGBENCH_CAUSALJUDGEMENT = "Bigbench_Causal_Judgement"
    BIGBENCH_MOVIE_RECOMENDATION = "Bigbench_Movie_Recommendation"
    BIGBENCH_FORMAL_FALLACIES = "Bigbench_Formal_Fallacies"
    BIGBENCH_DisambiguationQA = "Bigbench_DisambiguationQA"
    BIGBENCH_SNARKS = "Bigbench_Snarks"
    BIGBENCH_SPORTS = "Bigbench_Sports"
    BIGBENCH_GEOMETRIC_SHAPES = "Bigbench_Geometric_Shapes"
    BIGBENCH_PENGUINS = "Bigbench_Penguins"
    BIGBENCH_RUIN_NAMES = "Bigbench_Ruin_Names"
    BIGBENCH_TEMPORAL_SEQUENCES = "Bigbench_Temporal_Sequences"
    
    MATH_500 = "Math_500"
        
    @property
    def is_big_bench(self) -> bool:
        return self in [DatasetType.BIGBENCH_DATE, DatasetType.BIGBENCH_CAUSALJUDGEMENT,
                        DatasetType.BIGBENCH_MOVIE_RECOMENDATION, DatasetType.BIGBENCH_FORMAL_FALLACIES,
                        DatasetType.BIGBENCH_DisambiguationQA, DatasetType.BIGBENCH_SNARKS,
                        DatasetType.BIGBENCH_SPORTS, DatasetType.BIGBENCH_GEOMETRIC_SHAPES,
                        DatasetType.BIGBENCH_PENGUINS, DatasetType.BIGBENCH_RUIN_NAMES,
                        DatasetType.BIGBENCH_TEMPORAL_SEQUENCES]
    
    @property
    def is_output_number(self) -> bool:
        return self in [
            DatasetType.GSM8K, DatasetType.SVAMP,
            DatasetType.MATH_500
        ]
    
    @property
    def is_output_choice(self) -> bool:
        return self in [
                DatasetType.AQUA, DatasetType.CommonsenseQA, DatasetType.SVAMP,
            ] or self.is_big_bench



class DatasetEntry(BaseModel):
    answer: str
    question: str


class Dataset(BaseModel):
    data: List[DatasetEntry]
    datatype: DatasetType
    system_prompt: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i) -> DatasetEntry:
        assert 0<= i < len(self)
        return self.data[i]

    def __iter__(self):
        for i in range(len(self)):
            yield i, self.data[i]
