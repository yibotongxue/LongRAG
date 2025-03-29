from openai import OpenAI
from utils.load_data_util import load_json_file
import time
from utils.base_inference import BaseInference
from typing import List, Tuple, override

DEEPSEEK_API_KEY = ""

class DeepseekInference(BaseInference):
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    @override
    def _predict(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 1000)
        retry = kwargs.get("retry", 3)
        delay = kwargs.get("delay", 5)
        message_text=[
            {"role": "system", "content": prompt}
        ]
        response = None
        for _ in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=message_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=12345,
                    stream=False
                )
                break
            except:
                time.sleep(delay)
        response = self._post_process(response.choices[0].message.content)
        return response

    @override
    def predict_close_book(self, question: str, demo_file_path: str, num_demo: int=16) -> str:
        demo = load_json_file(demo_file_path)
        prompt = ("Here are some examples of questions and their corresponding answer, each with a 'Question' field and an 'Answer' field. "
                  "Answer the question directly and don't output other thing. ")
        for item in demo[:num_demo]:
            prompt += f"Question: {item['question']} Answer: {item['short_answers'][0]}\n"
        prompt += f"Question: {question} Answer: "
        answer = self._predict(prompt)
        return answer

    @override
    def predict_nq(self, context: str, question: str, titles: List[str]) -> Tuple[str, str]:
        titles = ['"' + title + '"' for title in titles]
        prompt = (f"Go through the following context and then extract the answer of the question from the context. "
                  f"Answer the question directly. Your answer should be very concise. "
                  f"The context is a list of Wikipedia documents, ordered by title: {titles}. "
                  f"Each Wikipedia document contains a 'title' field and a 'text' field. "
                  f"The context is: {context}. "
                  f"The question: {question}. ")
        long_answer = self._predict(prompt)
        short_answer = self._extract_answer(question, long_answer)
        return long_answer, short_answer

    @override
    def predict_hotpotqa(self, context: str, question: str, titles: List[str]) -> Tuple[str, str]:
        prompt = (f"Go through the following context and then answer the question "
                  f"The context is a list of Wikipedia documents titled: {titles}. "
                  f"There are two types of questions: comparison questions, which require a yes or no answer or a selection from two candidates, "
                  f"and general questions, which demand a concise response. "
                  f"The context is: {context}. "
                  f"Find the useful documents from the context, then answer the question: {question}."
                  f"For general questions, you should use the exact words from the context as the answer to avoid ambiguity. "
                  f"Answer the question directly and don't output other thing.  ")
        long_answer = self._predict(prompt)
        short_answer = self._extract_answer(question, long_answer)
        return long_answer, short_answer
