from __future__ import annotations

import os
import json
import base64
import uuid
import aiohttp

from dataclasses import dataclass
from typing import Optional

from livekit.agents.tts.tts import (
    TTS as BaseTTS,
    TTSCapabilities,
    ChunkedStream as _BaseChunkedStream,
    AudioEmitter,
)
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents._exceptions import APIError  # <-- ИСПРАВЛЕНО: корректный импорт


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    sample_rate: int = 24_000
    follow_redirects: bool = True


class TTS(BaseTTS):
    """Яндекс SpeechKit TTS плагин для LiveKit Agents"""

    API_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"

    def __init__(
        self,
        *,
        voice: str = "masha",
        sample_rate: int = 16_000,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        api_key = os.getenv("YANDEX_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения YANDEX_API_KEY не установлена")

        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            api_key=api_key,
            voice=voice,
            sample_rate=sample_rate,
        )
        # берём общую сессию LiveKit, если не передали свою
        from livekit.agents import utils as _utils  # локальный импорт, чтобы не тянуть при инициализации

        self._session = http_session or _utils.http_context.http_session()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _BaseChunkedStream:
        return _YandexRESTChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        pass


class _YandexRESTChunkedStream(_BaseChunkedStream):
    async def _run(self, output_emitter: AudioEmitter) -> None:
        opts = self._tts._opts  # type: ignore[attr-defined]
        session: aiohttp.ClientSession = self._tts._session  # type: ignore[attr-defined]

        request_id = uuid.uuid4().hex
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=200,
            stream=False,
        )

        payload = {
            "text": self.input_text,
            "hints": [{"voice": opts.voice}],
            "outputAudioSpec": {
                "rawAudio": {
                    "audioEncoding": "LINEAR16_PCM",
                    "sampleRateHertz": str(opts.sample_rate),
                }
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {opts.api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=self._conn_options.timeout)

        async with session.post(
            TTS.API_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
            allow_redirects=opts.follow_redirects,
        ) as resp:
            if resp.status != 200:
                try:
                    err_text = await resp.text()
                except Exception:
                    err_text = f"HTTP {resp.status}"
                # ИСПРАВЛЕНО: используем APIError
                raise APIError(f"Yandex TTS error (status {resp.status}): {err_text}")

            # читаем NDJSON поток без readline() (чтобы избежать ValueError: Chunk too big)
            buf = b""
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                buf += chunk

                while True:
                    nl_pos = buf.find(b"\n")
                    if nl_pos == -1:
                        break
                    line = buf[:nl_pos].rstrip(b"\r")
                    buf = buf[nl_pos + 1:]

                    if not line:
                        continue

                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    # возможная обёртка {"result": {...}}
                    if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict):
                        obj = obj["result"]

                    # достаём audioChunk.data
                    data_b64 = None
                    if isinstance(obj, dict):
                        ac = obj.get("audioChunk")
                        if isinstance(ac, dict):
                            data_b64 = ac.get("data")

                    if data_b64:
                        try:
                            raw = base64.b64decode(data_b64)
                        except Exception:
                            raw = b""
                        if raw:
                            output_emitter.push(raw)

            # обработать хвост без завершающего \n
            tail = buf.strip()
            if tail:
                try:
                    obj = json.loads(tail.decode("utf-8"))
                    if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict):
                        obj = obj["result"]
                    ac = obj.get("audioChunk") if isinstance(obj, dict) else None
                    if isinstance(ac, dict):
                        data_b64 = ac.get("data")
                        if data_b64:
                            raw = base64.b64decode(data_b64)
                            if raw:
                                output_emitter.push(raw)
                except Exception:
                    pass

            output_emitter.flush()
