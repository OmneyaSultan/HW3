import pickle
import argparse
from bm25 import *
from tfidf import *


def get_arguments():
    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", help="retriever model: what retriever to use", default="bm25")
    parser.add_argument("-i",  help="inputfile: the name/path of the file to index; it has to be read one text per line", default="test_data.txt")
    parser.add_argument("-n", help="index_name: the name/path of the index (you should write it on disk)", default="default.index")

    return parser.parse_args()



if __name__ == "__main__":
    args = get_arguments()

    if "bm25" in args.m:
        model = BM25(model_file=args.m)
    else:
        ## TODO Add any other models you wish to evaluate
        model = TFIDF(model_file=args.n)
        

    print(f"Running TF-IDF Indexing on file: {args.i}")  
    index = model.index(args.i)
    print(f"Indexing result: {index}")  

    print("TF-IDF Indexing Output:", index)
    results = model.search("what can I grow in the garden that's rewarding", k = 3)
    ## Save the index
    with open(args.n, "wb") as file:
        pickle.dump(index, file)
