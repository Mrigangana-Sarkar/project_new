from fastapi import APIRouter, Depends, Request
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.utils.auth import authenticated

chat_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChatBody(BaseModel):
    class ContextFilter(BaseModel):
        docs_ids: list[str] | None = None

    class Message(BaseModel):
        role: str
        content: str

    messages: list[Message]
    use_context: bool = False
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rapper. Always answer with a rap.",
                        },
                        {
                            "role": "user",
                            "content": "How do you fry an egg?",
                        },
                    ],
                    "stream": False,
                    "use_context": True,
                    "include_sources": True,
                    "context_filter": {
                        "docs_ids": ["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]
                    },
                }
            ]
        }
    }


@chat_router.post(
    "/chat/completions",
    response_model=None,
    responses={200: {"description": "Chat completion response"}},
    tags=["Contextual Completions"],
)
def chat_completion(
    request: Request, body: ChatBody
) -> dict | StreamingResponse:
    """Chat completion endpoint using LLama."""
    service = request.state.injector.get(ChatService)
    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    if body.stream:
        completion_gen = service.stream_chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
        )
        def stream():
            for chunk in completion_gen.response:
                yield f"data: {chunk}\n\n"
            if body.include_sources:
                yield f"data: [SOURCES] {completion_gen.sources}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        completion = service.chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
        )
        result = {"content": completion.response}
        if body.include_sources:
            result["sources"] = completion.sources
        return result
