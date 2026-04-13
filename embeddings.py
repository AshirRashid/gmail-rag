from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


class BGEEmbeddings(EmbeddingFunction):
    """
    Thin wrapper around SentenceTransformer for use as a ChromaDB
    EmbeddingFunction. Returns L2-normalised embeddings (cosine-ready).
    """

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return (
            self._model
            .encode(input, normalize_embeddings=True, show_progress_bar=False)
            .tolist()
        )
