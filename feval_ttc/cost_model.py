from .llm_interface import LLMType


class ModelCost(object):
    """This is a simplified monetary cost model the is computeing query cost as 
    <dollar_cost> = 1e-6 *( C1*<input_tokens> + C2*<output_tokens> )
    where C1 and C2 are input and output token costs in USD per million tokens.
    This model is made purely for benchmarking purposes.
    It does not account for caching, processing costs or price updates, thus the actual number may vary.
    The costs are recorded as of 02/06/2025
    """
    def __init__(self, llm_type: LLMType):
        match llm_type:
            case LLMType.DeepSeekV3:
                self.input_cost = 0.50
                self.output_cost = 1.50     
            case LLMType.LLaMA1B32:
                self.input_cost = 0.005
                self.output_cost = 0.01
            case LLMType.LLaMA3B32:
                self.input_cost = 0.01
                self.output_cost = 0.02
            case LLMType.LLaMA8B31:
                self.input_cost = 0.02
                self.output_cost = 0.06
            case LLMType.LLaMA70B33:
                self.input_cost = 0.13
                self.output_cost = 0.40
            case LLMType.LLaMA405B31:
                self.input_cost = 1.00
                self.output_cost = 3.00
            case LLMType.Qwen1B25:
                self.input_cost = 0.02
                self.output_cost = 0.06
            case LLMType.Qwen32B25:
                self.input_cost = 0.06
                self.output_cost = 0.20
            case LLMType.Qwen72B25:
                self.input_cost = 0.13
                self.output_cost = 0.40
            case LLMType.Mixtral8x7B:
                self.input_cost = 0.08
                self.output_cost = 0.24
            case LLMType.Mixtral8x22B:
                self.input_cost = 0.40
                self.output_cost = 1.20
            case LLMType.GPT3:
                self.input_cost = 0.50
                self.output_cost = 1.50
            case LLMType.GPT4oMINI:
                self.input_cost = 0.15
                self.output_cost = 0.60
            case LLMType.o3MINI:
                self.input_cost = 1.10
                self.output_cost = 4.40
            case _:
                raise ValueError(f"unsupported LLM type: {llm_type}")
        self.llm_type = llm_type
        
    def input_dollar_cost(self, text: str) -> float:
        return 1e-6 * self.input_cost * self.token_cost(text)
    
    def output_dollar_cost(self, text: str) -> float:
        return 1e-6 * self.output_cost * self.token_cost(text)
    
    def token_cost(self, text):
        # we are using an OpenAI approximation of the token count for the sake of speed.
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # 1000 words ~= 1365 tokens
        # the practical difference between this approximation and tiktoken is typically within 1-10 tokens over 1000 tokens
        return int(round(1.365 * len(text.split())))
