import argparse
from pathlib import Path
from retrievers.bm25 import BM25
from retrievers.faiss import FAISSIndex
from retrievers.document_loader import load_documents

def load_bm25(args):
  print("Loading BM25")
  retriever_config = {
    "k1": args.bm25_k1,
    "b": args.bm25_b,
    "epsilon": args.bm25_epsilon,
    "chunk_size": args.chunk_size
  }
  return BM25(**retriever_config)

def load_faiss(args):
  print("Loading Faiss")
  from retrievers.embedder import SnowFlakeArcticEmbedderLV2
  return FAISSIndex(SnowFlakeArcticEmbedderLV2(), chunk_size=args.chunk_size)


def get_arguments():
    parser = argparse.ArgumentParser(description="Given a model name and text, index the text")
    parser.add_argument("-m", help="retriever model: what retriever to use", default="bm25")
    parser.add_argument("-i",  help="inputfile: the name/path of the file to index", default="./datasets/retrieval_texts.txt")
    parser.add_argument("-n", help="index_name: the name/path of the index (you should write it on disk)")
    parser.add_argument("--sample", action="store_true", help="Sample only subset of available context docs", default=False)
    parser.add_argument("--chunk-size", help="Chunk size used to chunk documents before indexing", default=256, type=int)
    parser.add_argument("--bm25-k1", help="Controls k1 param for bm25", default=1.5)
    parser.add_argument("--bm25-b", help="Controls b param for bm25", default=0.75)
    parser.add_argument("--bm25-epsilon", help="Controls epsilon param for bm25", default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
  load_lut = {
    "bm25": load_bm25,
    "faiss": load_faiss
  }
  args = get_arguments()
  if not Path(args.i).exists():
    raise Exception("No input file found")
  docs = load_documents(args.i)
  if not args.m in load_lut:
    raise Exception(f"Invalid retriever, select from {load_lut.keys()}")
  retriever = load_lut[args.m](args)
  print("Starting index")
  if args.sample:
    docs = docs[:10]
  retriever.index(docs)
  retriever.store(args.n)
  





