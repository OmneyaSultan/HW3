from RetrievalModel import *
import numpy as np
import math
import collections
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class BM25(RetrievalModel):
    def __init__(self, model_file, b=0.75, k=1.2):
        self.b = b
        self.k = k
        self.term_doc_freqs = []
        self.doc_ids = []
        self.relative_doc_lens = []
        self.avg_num_words_per_doc = None
        self.inverted_index = collections.defaultdict(dict)  # Stores term frequencies per doc
        self.doc_lengths = {}  # Stores doc lengths
        self.total_docs = 0
        self.full_docs = collections.defaultdict(str)  # ✅ Stores full document text
        super().__init__(model_file)

    def index(self, input_file):
        """
        Train the BM25 retriever on the input dataset.
        :param input_file: path to training file with a text and a label per line
        """
        doc_store = collections.defaultdict(str)

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue  # ✅ Skip invalid lines
                doc_id, text = parts
                doc_store[doc_id] += " " + text  # ✅ Append text for multi-line docs
        for doc_id, text in doc_store.items():
            text = text.strip()
            self.full_docs[doc_id] = text
            words = [word for word in re.findall(r"\b\w+\b", text.lower()) if word not in ENGLISH_STOP_WORDS]
            # ✅ DEBUG: Print tokenized words for each document
            print(f"📝 Doc {doc_id} Tokens:", words)
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = len(words)
            self.total_docs += 1
            term_freqs = collections.Counter(words)
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq

        self.avg_num_words_per_doc = sum(self.doc_lengths.values()) / self.total_docs

        return {
            "inverted_index": self.inverted_index,
            "doc_lengths": self.doc_lengths,
            "avg_num_words_per_doc": self.avg_num_words_per_doc,
            "doc_ids": self.doc_ids,
            "full_docs": self.full_docs,  # ✅ Store full document text
        }

    def search(self, query, k):
        """
        Perform retrieval using BM25 scoring.
        :param query: query string
        :param k: number of results to retrieve
        """
        query_terms = re.findall(r"\b\w+\b", query.lower())  # ✅ Extract clean words
        scores = {doc_id: 0 for doc_id in self.doc_ids}  # Initialize scores
        for term in query_terms:
            if term not in self.inverted_index:
                continue  # Skip terms not in corpus
            # IDF calculation
            num_docs_with_term = len(self.inverted_index[term])
            idf = math.log((self.total_docs + 1) / (num_docs_with_term + 0.5) + 1)
            for doc_id, freq in self.inverted_index[term].items():
                # BM25 formula
                doc_length = self.doc_lengths[doc_id]
                term_score = (freq * (self.k + 1)) / (freq + self.k * (1 - self.b + self.b * (doc_length / self.avg_num_words_per_doc)))
                scores[doc_id] += idf * term_score
        # Sort and return top-k documents
        top_k_docs = sorted(scores, key=scores.get, reverse=True)[:k]
        return [(doc_id, self.full_docs[doc_id]) for doc_id in top_k_docs]
