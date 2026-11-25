
import logging
from functools import lru_cache

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD MODEL LAZILY
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load the embedding model only when an embedding
    is requested. Reuse the model after first loading.
    """

    logger.info("Loading embedding model: %s", MODEL_NAME)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    logger.info("Embedding model loaded successfully.")

    return model


# --------------------------------------------------
# GENERATE SINGLE EMBEDDING
# --------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """
    Convert one text into a normalized 384-dimensional vector.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")

    model = load_embedding_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.tolist()


# --------------------------------------------------
# GENERATE BATCH EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Generate embeddings for multiple transcript chunks.
    """

    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Every input text must be a non-empty string."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()
