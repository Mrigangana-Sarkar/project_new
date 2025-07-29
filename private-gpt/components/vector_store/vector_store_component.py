import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)


from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

class ContextFilter:
     docs_ids:list[str] | None = None

def _doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> MetadataFilters:
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        for doc_id in context_filter.docs_ids:
            filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))

    return filters


@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store: BasePydanticVectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from llama_index.vector_stores.qdrant import (  # type: ignore
                QdrantVectorStore,
            )
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as e:
                raise ImportError(
                    "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
                ) from e

        if settings.qdrant is None:
                logger.info(
                        "Qdrant config not found. Using default settings."
                        "Trying to connect to Qdrant at localhost:6333."
                        )
                client = QdrantClient()
        else:
                client = QdrantClient(
                    **settings.qdrant.model_dump(exclude_none=True)
                )
        self.vector_store = typing.cast(
                        BasePydanticVectorStore,
                        QdrantVectorStore(
                            client=client,
                            collection_name="make_this_parameterizable_per_api_call",
                        ),  # TODO
                    )

    def get_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 5,
    ) -> VectorIndexRetriever:
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter)
                if self.settings.vectorstore.database != "qdrant"
                else None
            ),
        )

    def close(self) -> None:
        if hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
