from datasets import load_dataset
import re
from abc import ABC, abstractmethod
from typing import Tuple, List

class BaseInference(ABC):
    def __init__(self):
        pass

    def _post_process(self, text:str) -> str:
        match = re.search(r"(?i)(?<=\banswer:\s).*", text)
        if match:
            return match.group(0)
        else:
            return text

    @abstractmethod
    def _predict(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def predict_close_book(self, question: str, demo_file_path: str, num_demo: int=16) -> str:
        pass

    @abstractmethod
    def predict_nq(self, context: str, question: str, titles: List[str]) -> Tuple[str, str]:
        pass

    @abstractmethod
    def predict_hotpotqa(self, context: str, question: str, titles: List[str]) -> Tuple[str, str]:
        pass

    def __generate_demo_examples(self, num_demo: int=4) -> str:
        if num_demo == 0:
            return ""
        demo_data = load_dataset("TIGER-Lab/LongRAG", "answer_extract_example")['train']
        demo_prompt = "Here are some examples: "
        for item in demo_data.select(range(num_demo)):
            for answer in item["answers"]:
                demo_prompt += f"Question: {item["question"]}\nLong Answer: {item["long_answer"]}\nShort Answer: {answer}\n\n"
        return demo_prompt

    def _extract_answer(self, question: str, long_answer: str) -> str:
        prompt = "As an AI assistant, you have been provided with a question and its long answer. " \
                 "Your task is to derive a very concise short answer, extracting a substring from the given long answer. " \
                 "Short answer is typically an entity without any other redundant words." \
                 "It's important to ensure that the output short answer remains as simple as possible.\n\n"
        prompt += self.__generate_demo_examples(num_demo=8)
        prompt += f"Question: {question}\nLong Answer: {long_answer}\nShort Answer: "
        short_answer = self._predict(prompt)
        return short_answer
