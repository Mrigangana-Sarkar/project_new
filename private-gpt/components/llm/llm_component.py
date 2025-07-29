import logging
from injector import inject, singleton
from llama_index.core.llms import LLM
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer

from private_gpt.components.llm.prompt_helper import get_prompt_style
from private_gpt.paths import models_cache_path, models_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing LLaMA 3 via LlamaCPP...")

        if settings.llm.tokenizer:
            try:
                set_global_tokenizer(
                    AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=settings.llm.tokenizer,
                        cache_dir=str(models_cache_path),
                        token=settings.huggingface.access_token,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download tokenizer {settings.llm.tokenizer}: {e}"
                    " — falling back to default tokenizer."
                )

        try:
            from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LlamaCPP not installed. Use `poetry install --extras llms-llama-cpp`."
            ) from e

        prompt_style = get_prompt_style("llama3")

        self.llm = LlamaCPP(
            model_path=str(models_path / settings.llamacpp.llm_hf_model_file),
            temperature=settings.llm.temperature,
            max_new_tokens=settings.llm.max_new_tokens,
            context_window=settings.llm.context_window,
            callback_manager=LlamaIndexSettings.callback_manager,
            model_kwargs={
                "tfs_z": settings.llamacpp.tfs_z,
                "top_k": settings.llamacpp.top_k,
                "top_p": settings.llamacpp.top_p,
                "repeat_penalty": settings.llamacpp.repeat_penalty,
                "n_gpu_layers": 10,
                "offload_kqv": True,
            },
            messages_to_prompt=prompt_style.messages_to_prompt,
            completion_to_prompt=prompt_style.completion_to_prompt,
            verbose=True,
        )
