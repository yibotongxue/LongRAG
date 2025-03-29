import google.generativeai as genai
from utils.load_data_util import load_json_file
from utils.base_inference import BaseInference
from typing import Tuple, List, override


genai.configure(api_key="")

generation_config = {
    "temperature": 0.0,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


class GeminiInference(BaseInference):
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

    @override
    def _predict(self, prompt: str, **kwargs) -> str:
        chat_session = self.model.start_chat(
            history=[]
        )
        response = chat_session.send_message(prompt)
        return self._post_process(response.text)

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
