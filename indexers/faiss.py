import faiss
import numpy as np
import os

class FAISSIndex:
    def __init__(self, embedder):
        """
        Initialize with a provided embedder.
        The embedder should have an `encode()` method that converts a list of strings to a list/array of embeddings.
        """
        self.embedder = embedder  # external embedder, e.g., from sentence-transformers
        self.index = None         # FAISS index instance
        self.docs = []            # original documents
        self.embeddings = None    # document embeddings (numpy array)

    def Index(self, docs: list[str]):
        """
        Build the FAISS index for a list of documents.
        """
        self.docs = docs
        # Compute embeddings for the documents.
        # Assuming the embedder returns a list of vectors.
        vecs = self.embedder.embed_documents(docs)
        vecs = np.array(vecs).astype('float32')
        self.embeddings = vecs

        # Create a FAISS index. Here we use IndexFlatL2 for L2 (Euclidean) distance.
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def Search(self, query: str, top_k: int = 3):
        """
        Given a query string, return the top_k most similar documents along with their distances.
        """
        if self.index is None:
            raise ValueError("Index not built. Please call Index(docs) first.")

        # Embed the query (note: embedder.encode expects a list, hence [query])
        q_vec = self.embedder.embed_query(query)
        q_vec = np.array(q_vec).astype('float32')

        # Perform the search
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):
                results.append((self.docs[idx], dist))
        return results

    def Store(self, path: str):
        """
        Store the FAISS index, embeddings, and original documents to the specified directory.
        """
        if self.index is None:
            raise ValueError("Index not built. Nothing to store.")
        
        # Ensure the directory exists.
        os.makedirs(path, exist_ok=True)
        
        # Save the FAISS index.
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save the embeddings.
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        # Save the documents.
        with open(os.path.join(path, "docs.txt"), "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(doc + "\n")

    def Load(self, path: str):
        """
        Load the FAISS index, embeddings, and original documents from the specified directory.
        """
        # Load the FAISS index.
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load the embeddings.
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        
        # Load the documents.
        with open(os.path.join(path, "docs.txt"), "r", encoding="utf-8") as f:
            self.docs = f.read().splitlines()
