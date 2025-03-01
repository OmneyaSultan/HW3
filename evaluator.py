

import argparse
import unicodedata

def simplify_text(text):
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text


def compare(to_score, truth):
  correct = 0
  for answer, reference in zip(to_score, truth):
    reference_possibilities = reference.split('\t')
    if any(simplify_text(word.lower().strip()) in simplify_text(answer.lower().strip()) for word in reference_possibilities):
      correct += 1
    return correct / len(truth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, required=True, default = "datasets/results_old/dev_gemma2_7b_it_faiss_rr.answers.txt")
    parser.add_argument("-t", type=str, required=True, default = "datasets/answers.dev.txt")
    args = parser.parse_args()

    with open(args.p, "r") as f:
        to_score = f.readlines()

    with open(args.t, "r") as f:
        truth = f.readlines()[1:]

    assert len(to_score) == len(truth), "Predictions and truth must have the same number of lines"

    score = compare(to_score, truth)

    print(score)