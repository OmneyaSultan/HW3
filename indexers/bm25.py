import re
import pickle
from rank_bm25 import BM25Okapi
from document_loader import load_documents

class BM25:
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25, max_words=200):
        """
        Initialize the BM25 index with tuning parameters.
        
        Args:
            k1 (float): Term frequency scaling parameter.
            b (float): Document length scaling parameter.
            epsilon (float): Small constant to avoid zero division.
            max_words (int): Maximum number of words per chunk; longer texts will be recursively split.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.max_words = max_words
        
        # These lists will store:
        #   - chunks: text chunks after splitting
        #   - doc_ids: mapping each chunk back to its original document index
        #   - tokenized_chunks: token lists for each chunk used by BM25
        self.chunks = []
        self.doc_ids = []
        self.tokenized_chunks = []
        
        # The BM25 model will be built once index() is called.
        self.bm25 = None

    def index(self, documents):
        """
        Build the BM25 index from a list of documents.
        Each document is split into chunks (if needed) and tokenized.
        
        Args:
            documents (List[str]): A list of document texts.
        """
        # Reset current index
        self.chunks = []
        self.doc_ids = []
        self.tokenized_chunks = []
        
        for doc_id, doc in enumerate(documents):
            self.chunks.append(doc)
            self.doc_ids.append(doc_id)
            # Basic whitespace tokenization; feel free to replace with a more advanced tokenizer if needed.
            tokens = doc.split()
            self.tokenized_chunks.append(tokens)
        
        # Build the BM25 model using the tokenized chunks
        self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(self, query, n_results=10):
        """
        Search the BM25 index for a given query string.
        
        Args:
            query (str): The query text.
            n_results (int): Number of top results to return.
        
        Returns:
            List[Tuple[int, str, float]]: A list of tuples containing:
                - original document index,
                - the text chunk,
                - BM25 score.
        """
        if self.bm25 is None:
            raise ValueError("The index has not been built yet. Please call index() first.")
        
        # Tokenize the query (using simple whitespace splitting)
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get indices of the top scoring chunks
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        results = [(self.doc_ids[i], self.chunks[i], scores[i]) for i in top_indices]
        return results

    def save(self, file_path):
        """
        Save the BM25 index to a file via pickle.
        
        Args:
            file_path (str): The file path to save the index.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """
        Load a BM25 index from a pickle file.
        
        Args:
            file_path (str): The file path from which to load the index.
            
        Returns:
            BM25Index: The loaded BM25 index instance.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        """
        Custom pickle behavior: remove the live BM25 model instance 
        (which can be rebuilt) before pickling.
        """
        state = self.__dict__.copy()
        if "bm25" in state:
            del state["bm25"]
        return state

    def __setstate__(self, state):
        """
        Restore state from pickle and rebuild the BM25 model.
        """
        self.__dict__.update(state)
        if hasattr(self, "tokenized_chunks") and self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b, epsilon=self.epsilon)
        else:
            self.bm25 = None

# Example usage:
if __name__ == "__main__":
    docs = load_documents("./datasets/retrieval_texts.txt")
    
    # Initialize the BM25 index with custom parameters
    index = BM25(k1=1.2, b=0.75, epsilon=0.25, max_words=15)
    index.index(docs)
    
    # Perform a search
    results = index.search("What are Dhalias", n_results=5)
    for doc_id, chunk, score in results:
        print(f"Doc ID: {doc_id}, Score: {score:.4f}\nChunk: {chunk}\n")
    
    # Save the index to disk
    index.save("bm25_index.pkl")
    
    # Later, you can restore the index:
    loaded_index = BM25.load("bm25_index.pkl")
    loaded_results = loaded_index.search("document", n_results=3)
    for doc_id, chunk, score in loaded_results:
        print(f"Loaded - Doc ID: {doc_id}, Score: {score:.4f}\nChunk: {chunk}\n")

