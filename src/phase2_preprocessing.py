
import torch
import importlib


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def initialize_chunker():
    spacy = importlib.import_module("spacy")

    return spacy.load(
        "en_core_web_sm",
        disable=["ner", "tagger", "lemmatizer", "attribute_ruler"]
    )


def chunk_text(text: str, nlp_model) -> list[str]:
    doc = nlp_model(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


class MiniLMFeatureExtractor:

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        sentence_transformers = importlib.import_module(
            "sentence_transformers"
        )
        self.model = sentence_transformers.SentenceTransformer(
            MODEL_NAME,
            device=self.device
        )
        self.model.eval()
        self.embedding_dim = self.model.get_embedding_dimension()


    def get_embeddings(
        self,
        chunks: list[str]
    ) -> torch.Tensor:

        if not chunks:
            return torch.empty(
                (0, self.embedding_dim),
                device=self.device
            )

        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

        return embeddings.to(self.device)
