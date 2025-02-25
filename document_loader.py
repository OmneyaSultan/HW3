import re

def load_documents(path):
    with open(path) as f:
        text = f.read()
    pattern = r'"[0-9_]+\.txt\t.*?'
    matches = list(re.finditer(pattern, text))
    bounds = [(m1.start(), m2.start()) for m1, m2 in zip(matches, matches[1:])]
    documents = []
    for start, stop in bounds:
        document_content = text[start: stop]
        matches = re.findall(r"[0-9_]+\.txt\t", document_content)
        if len(matches) > 1:
            raise Exception("Too many matches")
        if len(matches) == 0:
            raise Exception("No document name found")
        document_id = matches[0]
        document_content = document_content.replace(document_id, "")
        documents.append(document_content)
    return documents


# Example usage:
if __name__ == "__main__":
    docs = load_documents("./datasets/retrieval_texts.txt")
    for doc in docs:
        print("ID:", doc["id"])
        print("Text snippet:", doc["text"][:100], "...")
        print("=" * 80)
