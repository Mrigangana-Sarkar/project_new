import logging

from injector import inject, singleton
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing HuggingFace embedding model from local path...")

        self.embedding_model = HuggingFaceEmbedding(
            model_name="models/embedding",  # local path to the downloaded Nomic model
            cache_folder=str(models_cache_path),
            trust_remote_code=settings.huggingface.trust_remote_code,
        )
