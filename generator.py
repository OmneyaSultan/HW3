import argparse
from pathlib import Path
import os
import json
from tqdm import tqdm

class RAG:
  def __init__(self, generator, retriever=None, reranker=None):
    self.retriever = retriever
    self.generator = generator
    self.retriever = retriever
    self.reranker = reranker

  def query(self, query):
    def unpack_result(result):
      docs = [entry[0] for entry in result]
      scores = [float(entry[1]) for entry in result]
      return docs, scores
  
    if self.retriever is not None:
      docs, scores = unpack_result(self.retriever.search(query))
    else:
      docs, scores = [], []
    if self.reranker is not None:
      docs, scores = unpack_result(self.reranker.rerank(query, docs))
  
    answer = self.generator.query(query, docs)
    return {"context": list(zip(docs, scores)), "answer": answer}

def get_arguments():
    parser = argparse.ArgumentParser(description="Generator")
    parser.add_argument('-r', help="name of the retriever model you used.", default="bm25")
    parser.add_argument("-n", help="the name of the index that you created with 'index.py'.", default="datasets/indexer/bm25/")
    parser.add_argument("-k", help="the number of documents to return in each retrieval run.", default=10, type=int)

    parser.add_argument("-m", help="type of model to use to generate: gemma2_2b, gemma2_7b_it, etc.", default="gemma2_2b")
    
    parser.add_argument("-i", help="path of the input file of questions, where each question is in the form: <text> for each newline", default="datasets/question.dev.txt")
    parser.add_argument("-o", help="path of the file where the answers should be written", default="datasets/results/dev") # Respect the naming convention for the model: make sure to name it *.answers.txt in your workplace otherwise the grading script will fail

    parser.add_argument("--no-retrieve", action="store_true", help="Disables retrieval and reranking", default=False)
    parser.add_argument("--rerank", action="store_true", help="Controls whether reranking is enabled with cross encoder", default=False)
    parser.add_argument("--sample", action="store_true", help="Controls whether a sample of evaluation dataset questions is tested", default=False)
    parser.add_argument("--rr-top-k", help="Number of context units retrieved by reranker", default=5, type=int)
    return parser.parse_args()
  
def load_bm25(args):
  from retrievers.bm25 import BM25
  print("Loading BM25")
  retriever = BM25.load(args.n)
  retriever.top_k = args.k
  return retriever

def load_faiss(args):
  print("Loading Faiss")
  from retrievers.faiss import FAISSIndex
  from retrievers.embedder import SnowFlakeArcticEmbedderLV2
  retriever = FAISSIndex(SnowFlakeArcticEmbedderLV2())
  retriever.load(args.n)
  retriever.top_k = args.k
  return retriever

def load_gemma2_2b():
  from generators.hf_generator import Gemma2_2b
  return Gemma2_2b()

def load_gemma2_7b_it():
  from generators.qhf_generator import Gemma2_7b_it
  return Gemma2_7b_it()


GENERATOR_LUT = {
  "gemma2_2b": load_gemma2_2b,
  "gemma2_7b_it": load_gemma2_7b_it
}

RETRIEVER_LUT = {
  "bm25": load_bm25,
  "faiss": load_faiss
}

def evaluate(flow, questions):
  print(f"Evaluating {len(questions)}")
  answers = []
  for question in tqdm(questions):
    result = flow.query(question)
    result["question"] = question
    answers.append(result)
  return answers

def save_answers(flow_name, results, path):
  path = Path(path)
  os.makedirs(path.parent, exist_ok=True)
  with open(path.parent / f"{flow_name}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
  # To conform to autograder
  with open(path.parent / f"{path.name}_{flow_name}.answers.txt", "w", encoding="utf-8") as f:
    for result in results:
      f.write(result['answer'].replace("\n", "\\n") + '\n')

def get_flow_name(args):
  if args.rerank:
    rerank = "_rr"
  else:
    rerank = ""
  if not args.no_retrieve:
    return f"{args.m}_{args.r}{rerank}"
  else:
    return f"{args.m}"

if __name__ == "__main__":
    args = get_arguments()
    if args.rerank and not args.no_retrieve:
      print("Loading Reranker")
      from retrievers.reranker import HFReranker
      reranker = HFReranker(top_k=args.rr_top_k)
    else:
      reranker = None
    if not args.no_retrieve:
      retriever = RETRIEVER_LUT[args.r](args)
    else:
      retriever = None
    generator = GENERATOR_LUT[args.m]()
    rag_flow = RAG(generator, retriever, reranker)
    questions = open(args.i, 'r', encoding='utf-8').readlines()[1:]
    if args.sample:
      questions = questions[:1]
    results = evaluate(rag_flow, questions)
    save_answers(get_flow_name(args), results, args.o)
