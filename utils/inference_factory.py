from utils.base_inference import BaseInference
from utils.claude_inference import ClaudeInference
from utils.gemini_inference import GeminiInference
from utils.gpt_inference import GPTInference
from utils.deepseek_inference import DeepseekInference

def create_inference(model_name: str) -> BaseInference:
    if model_name == "GPT-4o":
        return GPTInference()
    elif model_name == "Gemini":
        return GeminiInference()
    elif model_name == "Claude":
        return ClaudeInference()
    elif model_name == "DeepSeek":
        return DeepseekInference()
    raise ValueError(f"Unknown model name: {model_name}")
