from sentence_transformers import CrossEncoder


class Reranker:
  def __init__(self, name = "mixedbread-ai/mxbai-rerank-large-v1"):
    self.model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")
  def reranker(self, query, documents):
    results = self.model.rank(query, documents, return_documents=True, top_k=3)
  