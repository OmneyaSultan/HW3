import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class SnowFlakeArcticEmbedderLV2:
    def __init__(self, batch_size=8):
        self.model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size

    def embed_query(self, query):
        queries = [f'query: {query}']
        query_tokens = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=8192
        )
        query_tokens = query_tokens.to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**query_tokens)[0][:, 0]
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings.detach().cpu()

    def embed_documents(self, documents):
        doc_groups = []
        print(f"Embedding {len(documents)} documents...")
        for i in tqdm(range(0, len(documents), self.batch_size)):
            _documents = documents[i:i+self.batch_size]
            document_tokens = self.tokenizer(
                _documents,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=8192
            )
            document_tokens = document_tokens.to(self.device)
            with torch.no_grad():
                document_embeddings = self.model(**document_tokens)[0][:, 0]
                document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
                doc_groups.append(document_embeddings.cpu())
        return torch.cat(doc_groups)
