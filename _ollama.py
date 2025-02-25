# this is generic boilerplate for Ollama. 
# if you choose to implement your language model in some other way

# e.g. Huggingface Generator: https://huggingface.co/docs/transformers/en/main_classes/text_generation
#      or VLLM: https://blog.vllm.ai/2023/06/20/vllm.html

# make sure to overwrite the Generator class in the same way.

# Install Ollama here: https://ollama.com/download
# Then, make sure Ollama is running. If you have installed it correctly, just
# run `ollama serve` in your terminal. Either it will work, or it will fail (if Ollama is already running in the background.)

from GeneratorModel import *
from llama_index.llms.ollama import Ollama


class OllamaModel(GeneratorModel):
	def __init__(self, model_name=None):
		self.model = Ollama(model_name)  # Handle both cases
	def load_model(self):
		"""Dummy method to satisfy abstract class requirements."""
		return self 
	def query(self, context, question):
		prompt = f"Given the following context {context}\n\nAnswer the following question {question}"
		return self.model.complete(prompt, request_timeout=100000)
	

