from RetrievalModel import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import collections


class TFIDF(RetrievalModel):
    def __init__(self, model_file):
        super().__init__(model_file)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self.doc_lengths = {}
        self.avg_num_words_per_doc = None
        self.full_docs = collections.defaultdict(str)

    def index(self, input_file):
        """
        Train the TF-IDF model and save the index.
        :param input_file: path to training file with a text and a label per line
        """
        print(f"📂 Reading input file: {input_file}")  # ✅ Debugging

        doc_store = collections.defaultdict(str)

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue  # ✅ Skip invalid lines
                doc_id, text = parts
                doc_store[doc_id] += " " + text  # ✅ Append text for multi-line docs

        docs = []
        for doc_id, text in doc_store.items():
            text = text.strip()
            self.full_docs[doc_id] = text
            self.doc_ids.append(doc_id)
            docs.append(text)

        print(f"📊 Number of documents loaded: {len(docs)}")  # ✅ Debug print

        if not docs:
            print("❌ Error: No documents found in the input file.")
            return {}

        try:
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)  # ✅ Fit TF-IDF model
            print("✅ TF-IDF Model Trained Successfully")
        except Exception as e:
            print(f"❌ Error during TF-IDF training: {e}")
            return {}

        self.avg_num_words_per_doc = sum(len(doc.split()) for doc in docs) / len(docs)
        self.doc_lengths = {doc_id: len(docs[int(idx)].split()) for idx, doc_id in enumerate(self.doc_ids)}

        tfidf_index = {
            "inverted_index": self.tfidf_matrix,
            "doc_lengths": self.doc_lengths,
            "avg_num_words_per_doc": self.avg_num_words_per_doc,
            "doc_ids": self.doc_ids,
            "vectorizer": self.vectorizer,
            "full_docs": self.full_docs,  # ✅ Store full document text
        }

        print(f"✅ Returning TF-IDF Index: {tfidf_index.keys()}")  # ✅ Debugging print
        return tfidf_index  # ✅ Ensure dictionary is returned

    def search(self, query, k):
        """
        Perform retrieval using TF-IDF ranking.
        :param query: query string
        :param k: number of results to retrieve
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            print("❌ Error: TF-IDF model is not trained.")
            return []

        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray().flatten()
        ranked_indices = np.argsort(scores)[::-1][:k]  # Sort in descending order

        return [(self.doc_ids[idx], self.full_docs[self.doc_ids[idx]]) for idx in ranked_indices]
