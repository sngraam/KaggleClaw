from __future__ import annotations
import logging
import traceback
from typing import AsyncIterator, Any
from openai import AsyncOpenAI
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)

logger = logging.getLogger("kaggleclaw.client")

HARMONY_TOKENS = {
    200002: "<|return|>",
    200003: "<|constrain|>",
    200005: "<|channel|>",
    200006: "<|start|>",
    200007: "<|end|>",
    200008: "<|message|>",
    200012: "<|call|>",
}

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, context_tokens: int = 32000):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.context_tokens = context_tokens
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

    async def completion_stream(self, messages: list[Message], temperature: float = 1.0) -> AsyncIterator[dict[str, Any]]:
        conversation = Conversation.from_messages(messages)
        prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        
        max_tokens = self.context_tokens - len(prompt_ids)
        if max_tokens < 500:
            raise RuntimeError(f"Context length limits reached. Prompt tokens: {len(prompt_ids)}")

        stream = await self.client.completions.create(
            model=self.model,
            prompt=prompt_ids,
            max_tokens=min(max_tokens, 32256),
            temperature=temperature,
            stream=True,
            logprobs=5,
            stop=None,
            extra_body={
                "stop_token_ids": self.stop_token_ids,
                "return_token_ids": True,
                "min_p": 0.02,
            },
        )

        token_buffer: list[int] = []
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice: continue
            
            text_delta = choice.text or ""
            token_ids = self._extract_token_ids(chunk, choice, text_delta)
            if not token_ids:
                if text_delta:
                    yield {"type": "text", "content": text_delta}
                continue

            token_buffer.extend(token_ids)
            try:
                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                if new_messages:
                    yield {"type": "harmony_messages", "messages": new_messages, "token_ids": token_ids}
            except Exception:
                continue

    def _extract_token_ids(self, chunk: Any, choice: Any, text_delta: str) -> list[int]:
        token_ids = getattr(choice, "token_ids", None) or \
                    getattr(chunk, "token_ids", None) or \
                    getattr(choice, "model_extra", {}).get("token_ids") or \
                    getattr(chunk, "model_extra", {}).get("token_ids")
        
        if not token_ids and text_delta:
            try:
                token_ids = self.encoding.encode(text_delta)
            except Exception:
                pass
        return token_ids or []
