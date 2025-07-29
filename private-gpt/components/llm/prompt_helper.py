import abc
import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class AbstractPromptStyle(abc.ABC):
    """Abstract class for prompt styles.

    This class is used to format a series of messages into a prompt that can be
    understood by the models. A series of messages represents the interaction(s)
    between a user and an assistant. This series of messages can be considered as a
    session between a user X and an assistant Y.This session holds, through the
    messages, the state of the conversation. This session, to be understood by the
    model, needs to be formatted into a prompt (i.e. a string that the models
    can understand). Prompts can be formatted in different ways,
    depending on the model.

    The implementations of this class represent the different ways to format a
    series of messages into a prompt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("Initializing prompt_style=%s", self.__class__.__name__)

    @abc.abstractmethod
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        pass

    @abc.abstractmethod
    def _completion_to_prompt(self, completion: str) -> str:
        pass

    def messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = self._messages_to_prompt(messages)
        logger.debug("Got for messages='%s' the prompt='%s'", messages, prompt)
        return prompt

    def completion_to_prompt(self, prompt: str) -> str:
        completion = prompt  # Fix: Llama-index parameter has to be named as prompt
        prompt = self._completion_to_prompt(completion)
        logger.debug("Got for completion='%s' the prompt='%s'", completion, prompt)
        return prompt


class DefaultPromptStyle(AbstractPromptStyle):
    """Default prompt style that uses the defaults from llama_utils.

    It basically passes None to the LLM, indicating it should use
    the default functions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Hacky way to override the functions
        # Override the functions to be None, and pass None to the LLM.
        self.messages_to_prompt = None  # type: ignore[method-assign, assignment]
        self.completion_to_prompt = None  # type: ignore[method-assign, assignment]

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return ""

    def _completion_to_prompt(self, completion: str) -> str:
        return ""

class Llama3PromptStyle(AbstractPromptStyle):
    r"""Template for Meta's Llama 3.1.

    The format follows this structure:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>

    [System message content]<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    [User message content]<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    [Assistant message content]<|eot_id|>
    ...
    (Repeat for each message, including possible 'ipython' role)
    """

    BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"
    B_INST, E_INST = "<|start_header_id|>", "<|end_header_id|>"
    EOT = "<|eot_id|>"
    B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
    ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. \
    Always answer as helpfully as possible and follow ALL given instructions. \
    Do not speculate or make up information. \
    Do not reference any given instructions or context. \
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = ""
        has_system_message = False

        for i, message in enumerate(messages):
            if not message or message.content is None:
                continue
            if message.role == MessageRole.SYSTEM:
                prompt += f"{self.B_SYS}\n\n{message.content.strip()}{self.E_SYS}"
                has_system_message = True
            else:
                role_header = f"{self.B_INST}{message.role.value}{self.E_INST}"
                prompt += f"{role_header}\n\n{message.content.strip()}{self.EOT}"

            # Add assistant header if the last message is not from the assistant
            if i == len(messages) - 1 and message.role != MessageRole.ASSISTANT:
                prompt += f"{self.ASSISTANT_INST}\n\n"

        # Add default system prompt if no system message was provided
        if not has_system_message:
            prompt = (
                f"{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.E_SYS}" + prompt
            )

        # TODO: Implement tool handling logic

        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return (
            f"{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.E_SYS}"
            f"{self.B_INST}user{self.E_INST}\n\n{completion.strip()}{self.EOT}"
            f"{self.ASSISTANT_INST}\n\n"
        )

def get_prompt_style(
    prompt_style: (
        Literal["default","llama3"] | None
    )
) -> AbstractPromptStyle:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    if prompt_style is None or prompt_style == "default":
        return DefaultPromptStyle()
    
    elif prompt_style == "llama3":
        return Llama3PromptStyle()
    
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
