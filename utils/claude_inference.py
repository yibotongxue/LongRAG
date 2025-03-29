import anthropic
from utils.load_data_util import load_json_file
from utils.base_inference import BaseInference
from typing import Tuple, List, override


class ClaudeInference(BaseInference):
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key="",
        )

    @override
    def _predict(self, prompt: str, **kwargs) -> str:
        system_prompt = kwargs.get("system_prompt", "")
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return self._post_process(message.content[0].text)

    @override
    def predict_close_book(self, question: str, demo_file_path: str, num_demo: int=16) -> str:
        demo = load_json_file(demo_file_path)
        system_prompt = ("Here are some examples of questions and their corresponding answer, each with a 'Question' field and an 'Answer' field. "
                         "Answer the question directly and don't output other thing. The answer should be very concise")
        for item in demo[:num_demo]:
            system_prompt += f"Question: {item['question']} Answer: {item['answer']}\n"
        prompt = f"Question: {question} Answer: "
        answer = self._predict(prompt, system_prompt=system_prompt)
        return answer

    @override
    def predict_nq(self, context: str, question: str, titles: List[str]) -> Tuple[str, str]:
        prompt = (f"Go through the following context and then extract the answer of the question from the context. "
                  f"The context is a list of Wikipedia documents, ordered by title: {titles}. "
                  f"Each Wikipedia document contains a title field and a text field. "
                  f"The context is: {context}. "
                  f"Find the useful documents from the context, then extract the answer to answer the question: {question}."
                  f"Answer the question directly. Your response should be very concise. ")
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
