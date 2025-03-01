import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig


class Gemma2_2b:
  def __init__(self):
    self.pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b",
    device="cuda",  # replace with "mps" to run on a Mac device
  )

  def query(self, query, documents):
    context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n".join(documents)
    context = "\n\n\n\n##### INFORMATION ##### \n\n\n\n" + context
    input_text = f"Given this information {context}, the answer to this question '{query}' is"
    outputs = self.pipe(input_text, max_new_tokens=128)
    response = outputs[0]["generated_text"]
    response = response.replace(input_text, "")
    return response


class Phi3Small128KInstruct():
  def __init__(self, max_tokens=500):
    self.generation_args = {
      "max_new_tokens": 500,
      "return_full_text": False,
      "temperature": 0.0,
      "do_sample": False,
    }
    model_id = "microsoft/Phi-3-small-128k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    assert torch.cuda.is_available(), "This model needs a GPU to run ..."
    device = torch.cuda.current_device()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
  def query(self, query, documents):
    context = "\n\n\n\n##### CONTEXT ##### \n\n\n\n".join(documents)
    context = "\n\n\n\n##### CONTEXT ##### \n\n\n\n" + context
    prompt = f"I will answer the following question with the context {context}"
    messages = [
      {"role": "assistant", "content": prompt},
      {"role": "role", "content": query},
    ]
    output = self.pipe(messages, **self.generation_args)
    return output[0]['generated_text']