import re
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import spacy
from pathlib import Path
import os

nlp = spacy.load('en_core_web_lg')

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


class BM25:
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25, top_k=10, chunk_size=256):
      print(f"Instantiated BM25 model with args: k1:{k1}, b:{b}, epsilon:{epsilon}")
      self.k1 = k1
      self.b = b
      self.epsilon = epsilon
      self.chunks = []
      self.doc_ids = []
      self.tokenized_chunks = []
      self.bm25 = None
      self.top_k = top_k
      self.chunk_size = chunk_size

    def index(self, documents):
        print(f"Chunking {len(documents)} @ {self.chunk_size} chunks")
        _docs = []
        for doc in documents:
          for i in range(0, len(doc), self.chunk_size):
            _d = doc[i:i+self.chunk_size].strip()
            if len(_d) > 0:
              _docs.append(_d)
        documents = _docs
        self.chunks = []
        self.doc_ids = []
        self.tokenized_chunks = []
        print(f"Indexing {len(documents)}")
        for doc_id, doc in tqdm(enumerate(documents), total=len(documents)):
            self.chunks.append(doc)
            self.doc_ids.append(doc_id)
            tokens = lemmatize_text(doc)
            self.tokenized_chunks.append(tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(self, query):
        print(f"Retrieveing {self.top_k} context docs for query {query}")
        if self.bm25 is None:
            raise ValueError("The index has not been built yet. Please call index() first.")
        
        query_tokens = lemmatize_text(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        results = [(self.chunks[i], scores[i]) for i in top_indices]
        return results

    def store(self, file_path):
      file_path = Path(file_path)
      os.makedirs(file_path, exist_ok=True)
      file_path = file_path / "bm25.index"
      print(f"Storing BM25 Model at {file_path}")
      with open(file_path, "wb") as f:
          pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(f"{file_path}/bm25.index", "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "bm25" in state:
            del state["bm25"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "tokenized_chunks") and self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b, epsilon=self.epsilon)
        else:
            self.bm25 = None


