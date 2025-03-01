from sentence_transformers import CrossEncoder


class HFReranker:
  def __init__(self, name = "mixedbread-ai/mxbai-rerank-large-v1", batch_size=8, top_k=3):
    self.model = CrossEncoder(name)
    self.top_k = top_k
    self.batch_size = batch_size
  def rerank(self, query, documents):
    print(f"Reranking {self.top_k} context docs for query {query}")
    results = self.model.rank(query, documents, batch_size = self.batch_size, return_documents=True, top_k=self.top_k)
    results = [(entry["text"], entry["score"]) for entry in results]
    return results
