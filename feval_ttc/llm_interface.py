import random
from enum import StrEnum
from pydantic import BaseModel, computed_field, Field
from typing import List, Self, Optional, Literal


class LLMType(StrEnum):
    DeepSeekV3 = "deepseek-ai/DeepSeek-V3"
    LLaMA70B33 = "meta-llama/Llama-3.3-70B-Instruct"
    LLaMA8B31 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LLaMA405B31 = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    LLaMA3B32 = "meta-llama/Llama-3.2-3B-Instruct"
    LLaMA1B32 = "meta-llama/Llama-3.2-1B-Instruct"
    Qwen1B25 = "Qwen/Qwen2.5-1.5B-Instruct"
    Qwen32B25 = "Qwen/Qwen2.5-32B-Instruct"
    Qwen72B25 = "Qwen/Qwen2.5-72B-Instruct"
    
    Mixtral8x7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    Mixtral8x22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    #gpts
    GPT3 = "gpt-3.5-turbo"
    GPT4oMINI = "gpt-4o-mini"
    o3MINI = "o3-mini"
    
    @property
    def is_reasoning_model(self) -> bool:
        if self in [LLMType.o3MINI]:
            return True
        return False
    

class CoTMetadata(BaseModel):
    dollar_cost: float
    token_cost: float
    finish_reason: Literal["stop", "max_tokens", "content_filter", "run_cancelled", "run_expired", "run_failed"]


class CoT(BaseModel):
    raw_text: str
    answer: Optional[str] = None
    metadata: CoTMetadata


class LLMRequest(BaseModel):
    raw_text: str
    dollar_cost: float
    token_cost: float


class LLMResponse(BaseModel):
    cots: List[CoT]
    request: LLMRequest
    
    @computed_field
    @property
    def num_cots(self) -> int:
        return len(self.cots)
    
    def __len__(self) -> int:
        return self.num_cots
    
    @property
    def dollar_cost(self) -> float:
        return self.request.dollar_cost + sum(c.metadata.dollar_cost for c in self.cots)
    
    @property
    def token_cost(self) -> float:
        return self.request.token_cost + sum(c.metadata.token_cost for c in self.cots)
    
    @property
    def answers(self) -> List[str]:
        return [c.answer for c in self.cots]
    
    def to_json(self, filename):
        report_str = self.model_dump_json(indent=2)
        with open(filename, "w") as file:
            file.write(report_str)

    @classmethod
    def from_json(cls, filename: str) -> Self:
        with open(filename, "r") as file:
            data = file.read()
        return LLMResponse.model_validate_json(data)


class LLMConfig(BaseModel):
    name: LLMType
    temperature: float = Field(default=0.7, ge=0)
    max_tokens: int = Field(default=2048, gt=1)


class LLM(BaseModel):
    config: LLMConfig
    responses: List[LLMResponse]
    
    def __len__(self) -> int:
        return len(self.responses)
    
    def __call__(self, question_id, N=40, deterministic=False):
        assert N > 0
        
        response = self.responses[question_id]
        if N >= len(response.cots):
            return response.copy()
        if deterministic:
            idx = range(0, N)
        else:
            idx = random.sample(list(range(len(response.cots))), k=N)
        return LLMResponse(
            request=response.request.copy(),
            cots=[response.cots[i].copy() for i in idx]
        )
    
    