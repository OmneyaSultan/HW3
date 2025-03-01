import torch
from transformers import pipeline

class Gemma2_2b:
  def __init__(self):
    self.pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b",
    device="cuda",
  )

  def query(self, query, documents):
    if len(documents):
      context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n".join(documents)
      context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n" + context
      input_text = f"Given this information {context}, the answer to this question '{query}' is"
    else:
      input_text = f"The answer to this question '{query}' is "
    outputs = self.pipe(input_text, max_new_tokens=128)
    response = outputs[0]["generated_text"]
    response = response.replace(input_text, "")
    return response
