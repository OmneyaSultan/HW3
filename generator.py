import argparse
from _ollama import *
from bm25 import * 
from tfidf import *

def get_arguments():
    parser = argparse.ArgumentParser(description="Generator")
    parser.add_argument('-r', default="bm25", help="name of the retriever model you used.")
    parser.add_argument("-n", help="the name of the index that you created with 'index.py'.")
    parser.add_argument("-k", help="the number of documents to return in each retrieval run.")

    parser.add_argument("-p", default="ollama", help="Platform to use for the generator.")
    parser.add_argument("-m", help="type of model to use to generate: gemma2:2b, etc.")
    
    parser.add_argument("-i", help="path of the input file of questions, where each question is in the form: <text> for each newline")
    parser.add_argument("-o", help="path of the file where the answers should be written") # Respect the naming convention for the model: make sure to name it *.answers.txt in your workplace otherwise the grading script will fail

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.r == "bm25":
        retriever = BM25(args.n).load_model()
    else:
        retriever = TFIDF(args.n).load_model()

    if args.p == 'ollama':
        generator = OllamaModel(model_file=args.m)

    else:
        ## TODO Add any other models you wish to train
        generator = None

    answers = []
    questions = open(args.i).strip().splitlines()
    for q in questions:
        docs = retriever.search(q, args.k)
        answer = generator.query(docs, q)
        answers.append(answer)
    with open(args.o) as f:
        for a in answers:
            f.write(a)
            f.write('\n')






