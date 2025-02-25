

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, required=True)
    parser.add_argument("-t", type=str, required=True)
    args = parser.parse_args()

    with open(args.p, "r") as f:
        to_score = f.read().strip().split('\n')

    with open(args.t, "r") as f:
        truth = f.read().strip().split('\n')

    assert len(to_score) == len(truth), "Predictions and truth must have the same number of lines"

    correct = 0
    for pred, gold in zip(to_score, truth):
        for gold_item in gold.split('\t'):
            gold_item = gold_item.strip()
            if gold_item == '':
                continue

            if gold_item in pred:
                correct += 1
                break

    print(f"{correct / len(to_score)}")