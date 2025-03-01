# import argparse
# from pathlib import Path

# from HW3.document_loader import load_documents

# # from bm25 import BM25
# # from _ollama import OllamaModel
# # from document_loader import load_documents

# # def setup_faiss_index():


# # def get_arguments():
# #     # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
# #     parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
# #     parser.add_argument("-m", help="retriever model: what retriever to use", default="bm25")
# #     parser.add_argument("-i",  help="inputfile: the name/path of the file to index", default="./datasets/retrieval_texts.txt")
# #     parser.add_argument("-n", help="index_name: the name/path of the index (you should write it on disk)", default="bm25_index.pkl")

# #     return parser.parse_args()


# # if __name__ == "__main__":
# #     args = get_arguments()

# #     if not Path(args.n).exists():
# #         print("Loading Documents")
# #         docs = load_documents(args.i)
# #         print("Indexing")
# #         index = BM25(k1=1.2, b=0.75, epsilon=0.25, max_words=15)
# #         index.index(docs)
# #         print("Storing index")
# #         index.save(args.n)
    
# #     print("Loading index")
# #     retriever = BM25(k1=1.2, b=0.75, epsilon=0.25, max_words=15).load(args.n)
# #     print("Loading Model")
# #     model = OllamaModel("tinyllama")
# #     rag_loop(model, retriever)

