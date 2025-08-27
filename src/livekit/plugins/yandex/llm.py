"""Large‑language model integration for Yandex Cloud.

This module implements the LiveKit Agents LLM interface using the
foundation models provided by Yandex Cloud.  Due to limitations in
the official ``yandex_cloud_ml_sdk`` (which currently depends on
missing gRPC modules), this implementation communicates directly
with the Foundation Models API over HTTP.  It uses the OpenAI‑
compatible chat completions endpoint hosted at
``https://llm.api.cloud.yandex.net/v1/chat/completions``.  Partial
results are streamed back from the API when the ``stream`` option is
enabled, and these are mapped into LiveKit ``ChatChunk`` objects.

For details on the OpenAI compatibility layer, see the Yandex Cloud
documentation【971554106318272†L135-L169】.  Each chunk of the streaming
response contains a ``delta`` with incremental content and optional
tool calls.  At the end of the stream a ``[DONE]`` marker is sent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

try:
    # Import the LiveKit LLM base class under an alias.  We alias the
    # imported ``LLM`` to ``BaseLLM`` to avoid a name collision with the
    # concrete provider class defined below.  Importing other types
    # unchanged preserves their original names.
    from livekit.agents.llm import (
        LLM as BaseLLM,
        LLMCapabilities,
        ChatChunk,
        ChatDelta,
        ChatMessage,
        LLMStream,
        ToolCall,
    )
except ImportError:  # pragma: no cover
    # Provide simple stubs when LiveKit is not installed.  The stub
    # classes mirror the interface of the real LiveKit classes but do
    # nothing.  These are aliased as ``BaseLLM`` etc. so that the
    # provider class defined below can still be named ``LLM`` without
    # colliding with the imported symbol.
    class BaseLLM:  # type: ignore[misc]
        def __init__(self, capabilities, *args, **kwargs):
            self.capabilities = capabilities

    class LLMCapabilities:  # type: ignore[misc]
        def __init__(self, streaming: bool) -> None:
            self.streaming = streaming

    class ChatDelta:  # type: ignore[misc]
        def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None) -> None:
            self.content = content
            self.tool_calls = tool_calls

    class ChatChunk:  # type: ignore[misc]
        def __init__(self, id: str, delta: ChatDelta, finish_reason: Optional[str] = None) -> None:
            self.id = id
            self.delta = delta
            self.finish_reason = finish_reason

    class ChatMessage:  # type: ignore[misc]
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    class LLMStream:  # type: ignore[misc]
        def __init__(self, chat_ctx, tools, conn_options) -> None:
            self.chat_ctx = chat_ctx
            self.tools = tools
            self.conn_options = conn_options

    class ToolCall:  # type: ignore[misc]
        def __init__(self, id: str, name: str, arguments: Dict[str, Any]) -> None:
            self.id = id
            self.name = name
            self.arguments = arguments

# NOTE: We intentionally do not import ``yandex_cloud_ml_sdk`` here
# because the official SDK currently depends on generated gRPC
# modules that are not yet published on PyPI.  Instead, this plugin
# uses direct HTTP calls to the Yandex Cloud Foundation Models API.
# See ``YandexLLMStream._run`` for the HTTP implementation.  Should
# the official SDK become self‑contained in a future release this
# import can be reinstated.
_IMPORT_ERROR = None


class LLM(BaseLLM):
    """LiveKit Agents LLM plugin using Yandex Cloud foundation models.

    Parameters
    ----------
    folder_id:
        Identifier of the Yandex Cloud folder containing the model.  If you
        are using a service account the folder ID is embedded in your
        credentials; otherwise you must specify it here.
    api_key:
        API key or IAM token used for authentication with Yandex Cloud.
    model_name:
        Name of the foundation model to invoke.  For the GPT‑style
        chat model use ``yandexgpt``.  Defaults to this value.
    temperature:
        Sampling temperature; controls the randomness of the output.  The
        default is ``0.5``.  Lower values make the output more
        deterministic.
    max_tokens:
        Maximum number of tokens to generate.  Defaults to 1024.  Note
        that Yandex Cloud may enforce its own limits.
    """

    def __init__(
        self,
        folder_id: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        model_name: str = "yandexgpt",
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> None:
        """Create a new Yandex large‑language model interface.

        Parameters
        ----------
        folder_id:
            Identifier of the Yandex Cloud folder.  If omitted the
            ``YANDEX_FOLDER_ID`` environment variable is used.  This
            value is required when authenticating with an API key.
        api_key:
            API key or IAM token used for authentication.  If not
            provided the ``YANDEX_API_KEY`` environment variable will
            be used.
        model_name:
            Name of the foundation model to use, such as ``yandexgpt``.
        temperature:
            Sampling temperature for the model.
        max_tokens:
            Maximum number of tokens to generate per response.
        """
        import os

        # Resolve credentials from environment if not supplied.
        if api_key is None:
            api_key = os.getenv("YANDEX_API_KEY")
        if folder_id is None:
            folder_id = os.getenv("YANDEX_FOLDER_ID")
        if not api_key:
            raise ValueError(
                "Yandex LLM requires an API key; set YANDEX_API_KEY or pass api_key explicitly"
            )
        if not folder_id:
            raise ValueError(
                "Yandex LLM requires a folder ID; set YANDEX_FOLDER_ID or pass folder_id explicitly"
            )
        capabilities = LLMCapabilities(streaming=True)
        # Initialise the base class using the aliased ``BaseLLM``.
        super().__init__(capabilities)
        self.folder_id = folder_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # When using the HTTP implementation no SDK client is created.  The
        # folder ID and API key are stored for constructing the request to
        # Yandex Cloud's OpenAI‑compatible chat endpoint.
        self._sdk = None  # not used

    def chat(
        self,
        chat_ctx,
        tools: Optional[List[Any]] = None,
        request_options: Optional[Any] = None,
    ) -> "YandexLLMStream":
        """Create a streaming chat session with the Yandex model.

        This method instantiates a :class:`YandexLLMStream` which will
        manage the conversion of the chat context to the format expected
        by Yandex Cloud and then stream responses back to LiveKit.
        """
        return YandexLLMStream(chat_ctx, self, tools, request_options)


class YandexLLMStream(LLMStream):
    """LLM stream responsible for performing a chat completion.

    Instances of this class are created by :meth:`YandexLLM.chat`.  The
    ``_run`` coroutine is executed by the LiveKit runtime and yields
    ``ChatChunk`` objects as the model produces output.
    """

    def __init__(self, chat_ctx, parent: YandexLLM, tools: Optional[List[Any]], conn_options: Optional[Any]) -> None:
        super().__init__(chat_ctx, tools, conn_options)
        self._parent = parent
        self.chat_ctx = chat_ctx
        self.tools = tools or []
        self.conn_options = conn_options

    async def _run(self, request_id: str, event_sender) -> None:
        """Perform the chat completion and forward partial results.

        This coroutine constructs the payload expected by the Yandex
        Cloud OpenAI‑compatible chat completions endpoint and issues
        an HTTP request with streaming enabled.  As each partial
        response arrives, it is converted into a LiveKit ``ChatChunk``
        and placed onto ``event_sender``.  At the end of the stream
        processing stops when a ``[DONE]`` marker is received.
        """
        import aiohttp
        import json

        parent = self._parent

        # Convert chat context messages to the OpenAI message format.  We
        # expect ``chat_ctx.messages`` to be an iterable of ChatMessage
        # objects.  Each message is represented as a dict with ``role``
        # and ``content`` fields.  Tools results are not included here.
        messages: List[Dict[str, Any]] = []
        for msg in getattr(self.chat_ctx, "messages", []):
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            messages.append({"role": role, "content": content})

        # Convert LiveKit tool definitions into the format expected by
        # the OpenAI API.  Each tool is expressed as a function with a
        # name, description and JSON schema for parameters.
        tools_payload: List[Dict[str, Any]] = []
        for tool in self.tools:
            fn: Dict[str, Any] = {}
            name = getattr(tool, "name", None)
            if not name:
                continue
            fn["name"] = name
            # optional description
            desc = getattr(tool, "description", None)
            if desc:
                fn["description"] = desc
            params = getattr(tool, "parameters", None)
            if params:
                fn["parameters"] = params
            tools_payload.append({"type": "function", "function": fn})

        # Build the request payload.  The model URI must include the
        # folder ID and the model name followed by ``/latest``.  See
        # Yandex Cloud OpenAI compatibility documentation【971554106318272†L135-L169】.
        payload: Dict[str, Any] = {
            "model": f"gpt://{parent.folder_id}/{parent.model_name}/latest",
            "messages": messages,
            "temperature": parent.temperature,
            "max_tokens": parent.max_tokens,
            "stream": True,
        }
        if tools_payload:
            payload["tools"] = tools_payload

        # Derive timeout from connection options if provided.  The
        # LiveKit connection options use seconds for timeouts.  If no
        # timeout is specified we rely on aiohttp's default.
        client_timeout = None
        if self.conn_options is not None and hasattr(self.conn_options, "timeout"):
            t = self.conn_options.timeout
            if t:
                client_timeout = aiohttp.ClientTimeout(total=t)

        # Construct HTTP headers.  Yandex Cloud requires an API key or
        # IAM token in the ``Authorization`` header.  When using an
        # API key the prefix must be ``Api-Key``.  We also include
        # ``x-folder-id`` for completeness although the folder ID is
        # embedded in the model URI.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {parent.api_key}",
            "x-folder-id": parent.folder_id,
        }

        url = "https://llm.api.cloud.yandex.net/v1/chat/completions"

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                # Raise for HTTP errors early to surface authentication or
                # input issues.
                resp.raise_for_status()
                # The response is streamed as Server‑Sent Events (SSE) where
                # each line starting with ``data:`` contains a JSON
                # payload.  We accumulate bytes and split on newlines to
                # extract these events.
                buffer = ""
                async for chunk in resp.content.iter_any():
                    text = chunk.decode("utf-8", errors="ignore")
                    buffer += text
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        # Streaming ends when the API sends ``[DONE]``
                        if data_str == "[DONE]":
                            return
                        try:
                            event = json.loads(data_str)
                        except Exception:
                            # Skip malformed JSON events
                            continue
                        choices = event.get("choices", [])
                        for choice in choices:
                            delta_dict = choice.get("delta", {}) or {}
                            finish_reason = choice.get("finish_reason")
                            content = delta_dict.get("content")
                            # Collect tool calls if present.  Yandex
                            # follows the OpenAI format where each tool
                            # call is a dict with ``id``, ``type`` and
                            # ``function`` containing ``name`` and
                            # ``arguments``.  Convert these to LiveKit
                            # ``ToolCall`` objects.  Arguments may be
                            # passed as a JSON string; attempt to
                            # deserialize into a Python object.
                            tool_calls_data = delta_dict.get("tool_calls") or []
                            tool_calls: List[ToolCall] = []
                            for tc in tool_calls_data:
                                call_id = tc.get("id", "")
                                func = tc.get("function", {}) or {}
                                name = func.get("name", "")
                                arguments = func.get("arguments")
                                args: Any = {}
                                if arguments is not None:
                                    # Yandex may provide arguments as a JSON
                                    # string; attempt to decode.
                                    if isinstance(arguments, str):
                                        try:
                                            args = json.loads(arguments)
                                        except Exception:
                                            args = arguments
                                    else:
                                        args = arguments
                                tool_calls.append(ToolCall(call_id, name, args))
                            # Skip empty deltas; only emit when there is
                            # new content or tool calls.
                            if content is None and not tool_calls:
                                continue
                            delta = ChatDelta(content=content, tool_calls=tool_calls if tool_calls else None)
                            chunk = ChatChunk(id=request_id, delta=delta, finish_reason=finish_reason)
                            await event_sender.put(chunk)
