# this is generic boilerplate for Ollama. 
# if you choose to implement your language model in some other way

# e.g. Huggingface Generator: https://huggingface.co/docs/transformers/en/main_classes/text_generation
#      or VLLM: https://blog.vllm.ai/2023/06/20/vllm.html

# make sure to overwrite the Generator class in the same way.

# Install Ollama here: https://ollama.com/download
# Then, make sure Ollama is running. If you have installed it correctly, just
# run `ollama serve` in your terminal. Either it will work, or it will fail (if Ollama is already running in the background.)

import ollama 
from GeneratorModel import *


class OllamaModel(GeneratorModel):
	def __init__(self, model_name=None, model_file=None):
		self.model_name = model_name if model_name else model_file  # Handle both cases
  
	def load_model(self):
		"""Dummy method to satisfy abstract class requirements."""
		return self 


	def query(self, retrieved_documents, question):
		retrieved_documents_str = '\n'.join(retrieved_documents)
		prompt = PROMPT.format(retrieved_documents=retrieved_documents_str, quesion=question)
		response = ollama.chat(model=self.model_name, messages=[{
			'role': 'user',
			'content': prompt,
			},
		])
		return response['message']['content']

