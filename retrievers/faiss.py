import faiss
import numpy as np
import os
import json

class FAISSIndex:
    def __init__(self, embedder, top_k=10, chunk_size=256):
        self.embedder = embedder  # external embedder, e.g., from sentence-transformers
        self._index = None        # FAISS index instance
        self.docs = []            # original documents
        self.embeddings = None    # document embeddings (numpy array)
        self.top_k = top_k
        self.chunk_size = chunk_size

    def index(self, docs: list[str]):
        print(f"Chunking {len(docs)} @ {self.chunk_size} chunks")
        _docs = []
        for doc in docs:
          for i in range(0, len(doc), self.chunk_size):
            _d = doc[i:i+self.chunk_size].strip()
            if len(_d) > 0:
              _docs.append(_d)
        docs = _docs
        print(f"Indexing {len(docs)} chunks")
        self.docs = docs
        vecs = self.embedder.embed_documents(docs)
        vecs = vecs.cpu()
        vecs = np.array(vecs).astype('float32')
        self.embeddings = vecs

        dim = vecs.shape[1]
        # Vectors are normalized so this is equivelent to exact cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)

    def search(self, query: str):
        print(f"Retrieveing {self.top_k} context docs for query {query}")
        if self._index is None:
            raise ValueError("Index not built. Please call Index(docs) first.")
        q_vec = self.embedder.embed_query(query)
        q_vec = q_vec.cpu()
        q_vec = np.array(q_vec).astype('float32')
        distances, indices = self._index.search(q_vec, self.top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):
                results.append((self.docs[idx], dist))
        return results

    def store(self, path: str):
        if self._index is None:
            raise ValueError("Index not built. Nothing to store.")
        print(f"Saving index to {path}")
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "faiss.index"))
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.docs, f, indent=4)

    def load(self, path: str):
        self._index = faiss.read_index(os.path.join(path, "faiss.index"))
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "docs.json"), "r", encoding="utf-8") as f:
            self.docs = json.load(f)
