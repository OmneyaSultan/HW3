from sentence_transformers import SentenceTransformer

class Embedder:
  def __init__(self, name='Snowflake/snowflake-arctic-embed-l-v2.0'):
    self.model = SentenceTransformer(name)
  def embed_query(self, query):
    query_embeddings = self.model.encode([query], prompt_name="query")
    return query_embeddings
  def embed_documents(self, documents):
    document_embeddings = self.model.encode(documents)
    return document_embeddings 

