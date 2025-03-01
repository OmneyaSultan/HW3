import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Gemma2_7b_it:
  def __init__(self):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
    self.model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-7b-it",
        quantization_config=quantization_config,
    )

  def query(self, query, documents):
    context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n".join(documents)
    context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n" + context
    input_text = f"Use this information: {context}\n to answer this question: {query}"
    input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = self.model.generate(**input_ids, max_new_tokens=64)
    outputs = self.tokenizer.decode(outputs[0])
    outputs = outputs.replace(input_text, "")
    return outputs

