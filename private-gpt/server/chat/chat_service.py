from dataclasses import dataclass
from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor


class ContextFilter(BaseModel):
    docs_ids: list[str] | None = None


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )

    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        if use_context:
            vector_index_retriever = self.vector_store_component.get_retriever(
                index=self.index,
                context_filter=context_filter,
                similarity_top_k=settings.rag.similarity_top_k,
            )
            node_postprocessors: list[BaseNodePostprocessor] = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ]
            if settings.rag.similarity_value:
                node_postprocessors.append(
                    SimilarityPostprocessor(
                        similarity_cutoff=settings.rag.similarity_value
                    )
                )

            if settings.rag.rerank.enabled:
                rerank_postprocessor = SentenceTransformerRerank(
                    model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
                )
                node_postprocessors.append(rerank_postprocessor)

            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                node_postprocessors=node_postprocessors,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                llm=self.llm_component.llm,
            )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = chat_engine_input.chat_history if chat_engine_input.chat_history else None

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        streaming_response = chat_engine.stream_chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        return CompletionGen(
            response=streaming_response.response_gen,
            sources=sources,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = chat_engine_input.chat_history if chat_engine_input.chat_history else None

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        return Completion(response=wrapped_response.response, sources=sources)
