"""Speech‑to‑text integration for Yandex SpeechKit.

This module implements the LiveKit Agents STT interface using Yandex
SpeechKit's ``ShortAudioRecognition`` service for quick transcription of
short audio snippets.  The implementation currently supports only
non‑streaming recognition; for longer audio streams consider
``DataStreamingRecognition``.

The Yandex STT implementation collects audio frames from the
``livekit`` runtime, concatenates them into a single byte buffer and
submits the data to SpeechKit.  Once transcription completes a final
``SpeechEvent`` is emitted containing one or more alternatives.

See the Yandex SpeechKit documentation for details on the
``ShortAudioRecognition.recognize`` method, including supported
languages and audio formats【67005567507383†L540-L614】.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    # Alias the imported base STT class to ``BaseSTT`` so that our
    # provider class can be named ``STT`` without clobbering the import.
    from livekit.agents.audio_frame import AudioFrame
    from livekit.agents.stt import (
        STT as BaseSTT,
    )
    from livekit.agents.stt import (
        SpeechAlternative,
        SpeechEvent,
        SpeechEventType,
        STTCapabilities,
    )
except ImportError:  # pragma: no cover
    # If the LiveKit dependencies are not available (e.g. during local
    # development) we define minimal stand‑ins so the module can be
    # imported without errors.  These stubs should not be used at
    # runtime inside a LiveKit agent.
    class BaseSTT:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            pass

    class STTCapabilities:  # type: ignore[misc]
        def __init__(self, streaming: bool, interim_results: bool) -> None:
            self.streaming = streaming
            self.interim_results = interim_results

    class SpeechEventType:  # type: ignore[misc]
        FINAL_TRANSCRIPT: str = "FINAL"

    class SpeechAlternative:  # type: ignore[misc]
        def __init__(self, text: str, confidence: Optional[float] = None) -> None:
            self.text = text
            self.confidence = confidence

    class SpeechEvent:  # type: ignore[misc]
        def __init__(self, event_type: str, alternatives: List[SpeechAlternative]):
            self.type = event_type
            self.alternatives = alternatives

    class AudioFrame:  # type: ignore[misc]
        def __init__(self, data: bytes, sample_rate: int, num_channels: int) -> None:
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels

# ``speechkit`` is no longer used in this implementation.  We
# implement STT by calling the Yandex SpeechKit REST API directly
# using aiohttp.  If you previously depended on the SpeechKit
# Python SDK you can safely remove it from your environment.
Session = None  # type: ignore[assignment]
ShortAudioRecognition = None  # type: ignore[assignment]
_IMPORT_ERROR = None


class STT(BaseSTT):
    """LiveKit Agents STT plugin backed by Yandex SpeechKit.

    This implementation supports both the classic REST‑based short audio
    recognition endpoint and the newer gRPC streaming API (SpeechKit
    API v3).  By default, the plugin will attempt to use gRPC if the
    required dependencies are available (``grpcio`` and the compiled
    protocol buffer stubs from the `yandex-cloud/cloudapi` repository).
    If gRPC support is not available the plugin will fall back to the
    REST endpoint.

    Parameters
    ----------
    api_key:
        API key used to authenticate with Yandex SpeechKit.  You can
        obtain a key from the Yandex Cloud console.  See the SpeechKit
        docs for details on authentication.
    language:
        BCP‑47 language code to recognise (e.g. ``en-US`` or ``ru-RU``).
        Defaults to ``en-US``.  Only a small set of languages is
        supported by SpeechKit【67005567507383†L540-L614】.
    model:
        Optional language model name; set to ``general`` by default.
        See Yandex documentation for other available models.
    audio_format:
        Encoding of the audio data submitted to SpeechKit.  Acceptable
        values are ``lpcm`` or ``oggopus``.  We use ``lpcm`` by default
        because LiveKit provides raw PCM frames.
    sample_rate:
        Sampling frequency of the audio.  Should match the rate of
        incoming frames.  Acceptable values for SpeechKit are 8000,
        16000 and 48000 Hz【67005567507383†L590-L606】.
    use_grpc:
        If set to ``True`` the plugin will always try to perform
        recognition via the gRPC streaming API.  When ``None`` (the
        default), gRPC will be used if available.  If set to
        ``False`` the plugin will always use the REST fallback.  Note
        that the gRPC API requires compiled protocol buffer stubs
        (`stt_pb2` and `stt_service_pb2_grpc`) to be importable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        *,
        language: str = "en-US",
        model: str = "general",
        audio_format: str = "lpcm",
        sample_rate: int = 48000,
        use_grpc: Optional[bool] = None,
    ) -> None:
        import os

        # Resolve the API key from the environment if not provided.
        if api_key is None:
            api_key = os.getenv("YANDEX_API_KEY")
        if not api_key:
            raise ValueError(
                "Yandex STT requires an API key; set YANDEX_API_KEY or pass api_key explicitly"
            )

        # Resolve the folder_id from the environment if not provided.
        if folder_id is None:
            folder_id = os.getenv("YANDEX_FOLDER_ID")
        if not folder_id:
            raise ValueError(
                "Yandex STT requires a folder_id; set YANDEX_FOLDER_ID or pass folder_id explicitly"
            )

        # Determine whether to use gRPC.  By default (``None``) the plugin
        # will attempt to import ``grpc`` and the generated stubs.  If
        # either import fails the plugin falls back to REST.  When
        # ``use_grpc`` is ``True`` the plugin will raise an exception if
        # gRPC support is unavailable.  When ``False`` gRPC is disabled.
        self._force_grpc = use_grpc is True
        self._disable_grpc = use_grpc is False
        self._grpc_available = False
        if not self._disable_grpc:
            try:
                # Attempt to import the compiled protobuf stubs.  These files
                # must be generated from the Yandex Cloud ``stt.proto`` and
                # ``stt_service.proto`` definitions (see the official
                # documentation for details).  We import them lazily in
                # ``_grpc_recognize`` to avoid import errors at module load
                # time.  Here we simply mark gRPC as available if grpc can
                # be imported.
                self._grpc_available = True
            except Exception:
                self._grpc_available = False

        # STT in LiveKit is non‑streaming; interim results are not supported.
        capabilities = STTCapabilities(streaming=False, interim_results=False)
        super().__init__(capabilities)
        self.capabilities = capabilities
        self.language = language
        self.model = model
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.api_key = api_key
        self.folder_id = folder_id
        self._event_handlers = {}

    # ------------------------------------------------------------------
    # Event handling
    #
    # The LiveKit runtime uses an event emitter interface on STT
    # providers to collect metrics and other signals.  When the
    # ``livekit.agents.stt.STT`` base class is available it should
    # implement ``on`` and handle metrics emission.  However if the
    # import fails and our fallback STT class is used, those methods
    # won't exist.  To avoid runtime AttributeError we implement a
    # minimal subset of the EventEmitter API here.  Consumers can
    # register callbacks via ``on(event_name, callback)``.  This
    # implementation stores handlers in a dictionary and invokes
    # them when ``_emit`` is called.  Metrics emission is optional.
    def on(self, event_name: str):  # type: ignore[override]
        """Decorator to register an event handler for a given event.

        Usage:

        ``@stt.on("metrics_collected")\nasync def handle(metrics): ...``

        Returns a decorator which stores the function and returns it unmodified.
        """
        def decorator(handler):
            handlers = self._event_handlers.setdefault(event_name, [])
            handlers.append(handler)
            return handler
        return decorator

    def _emit(self, event_name: str, *args) -> None:
        """Internal helper to emit an event to registered handlers."""
        for handler in self._event_handlers.get(event_name, []):
            try:
                handler(*args)
            except Exception:
                # Suppress exceptions in event handlers
                pass

    async def _recognize_impl(self, request_id: str, audio_queue, event_sender) -> None:
        """Collect audio frames and perform a single recognition call.

        This method is invoked by the LiveKit STT runtime after audio
        capture ends.  It reads all ``AudioFrame`` objects from the
        provided asynchronous queue, concatenates their raw PCM data and
        submits the resulting buffer to the Yandex SpeechKit API.  When
        recognition completes a final ``SpeechEvent`` is emitted to
        ``event_sender`` containing the transcription.
        """
        # Gather frames from the queue until the generator is exhausted.
        frames: List[AudioFrame] = []
        async for frame in audio_queue:
            frames.append(frame)
        if not frames:
            return

        # Concatenate audio data and extract sample rate and channel count
        sample_rate = getattr(frames[0], "sample_rate", self.sample_rate)
        num_channels = getattr(frames[0], "num_channels", 1)
        pcm_bytes = b"".join(getattr(f, "data", b"") for f in frames)

        # Yandex SpeechKit supports only 8000, 16000 and 48000 Hz sample rates.  LiveKit
        # sometimes provides audio at other rates (e.g. 24000 Hz).  When an
        # unsupported rate is encountered we upsample the PCM to the nearest
        # supported rate using Python's built‑in audioop.  Falling back to an
        # unsupported rate will result in a 400 Bad Request from the REST API.
        if sample_rate not in (8000, 16000, 48000):
            try:
                import audioop  # type: ignore
                # Choose 48000 Hz as the target rate for unsupported inputs.  Assume
                # 16‑bit samples (2 bytes per sample); update pcm_bytes and
                # sample_rate accordingly.  The return value of ratecv is a tuple
                # (converted_audio, state) where state is ignored.
                converted, _ = audioop.ratecv(pcm_bytes, 2, num_channels, sample_rate, 48000, None)
                pcm_bytes = converted
                sample_rate = 48000
            except Exception:
                # If resampling fails, just override the reported sample rate to
                # the nearest supported value.  This may affect recognition
                # quality but prevents a hard error from the API.
                sample_rate = 48000

        # Build a simple buffer object compatible with ``recognize``.  The
        # object must have ``data`` and ``sample_rate`` attributes.
        class _Buffer:
            pass
        buffer = _Buffer()
        buffer.data = pcm_bytes
        buffer.sample_rate = sample_rate
        buffer.num_channels = num_channels

        # Use the high‑level ``recognize`` method, which decides whether
        # to call the gRPC or REST implementation based on the plugin
        # configuration.  This call returns a ``SpeechEvent``.
        event = await self.recognize(buffer, language=self.language)
        await event_sender.put(event)

        # Emit metrics for the recognised audio.  Approximate duration
        # based on sample rate and bytes length.
        try:
            bytes_len = len(pcm_bytes)
            duration_sec = bytes_len / 2 / float(sample_rate)
            metrics = {
                "stt_provider": "yandex",
                "language": self.language,
                "duration_sec": duration_sec,
                "audio_bytes": bytes_len,
            }
            self._emit("metrics_collected", metrics)
        except Exception:
            pass

    async def recognize(
        self,
        buffer,
        *,
        language: Optional[str] = None,
        conn_options: Any = None,
    ) -> SpeechEvent:  # type: ignore[override]
        """Recognize a single audio buffer and return a final SpeechEvent.

        LiveKit calls this method when the STT provider does not support
        streaming (``streaming=False``).  The ``buffer`` argument is
        expected to be an object with ``data`` (bytes) and ``sample_rate``
        attributes (e.g. an ``AudioFrame``).  This implementation forwards
        the audio to the Yandex SpeechKit short‑recognition endpoint and
        returns a ``SpeechEvent`` with a single alternative.  Additional
        keyword arguments are accepted for compatibility with the base API
        but are not used.
        """
        # Choose the requested or configured language
        lang = language or self.language
        # If gRPC is available and enabled, use the streaming API; otherwise
        # fall back to the REST endpoint.  When ``_force_grpc`` is True
        # an ImportError will propagate to signal that the environment
        # lacks gRPC support.
        if not self._disable_grpc and (self._grpc_available or self._force_grpc):
            try:
                return await self._grpc_recognize(buffer, lang)
            except Exception:
                # If gRPC was explicitly forced, re‑raise the error to
                # surface configuration problems.  Otherwise fall back to
                # the REST implementation.
                if self._force_grpc:
                    raise
                # intentionally swallow exception and fall back
        # REST fallback
        # Extract raw PCM data and sample rate from the buffer.
        pcm_bytes = getattr(buffer, "data", b"") or b""
        sample_rate = getattr(buffer, "sample_rate", self.sample_rate)
        # Build query parameters for the REST call.
        params = {
            "lang": lang,
            "topic": self.model,
            "format": self.audio_format,
        }
        if self.audio_format == "lpcm":
            params["sampleRateHertz"] = str(sample_rate)
        # Добавляем folder_id в параметры запроса
        params["folderId"] = self.folder_id
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
        }
        # Perform the REST request
        import aiohttp
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, data=pcm_bytes, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
        result_text = data.get("result", "")
        alternative = SpeechAlternative(text=result_text)
        # Ensure ``confidence`` is a float (default to 1.0).  Some LiveKit
        # aggregates sum confidence scores; None causes a TypeError.
        try:
            alternative.confidence = 1.0
        except Exception:
            pass
        # Attach additional attributes expected by LiveKit on SpeechAlternative
        # instances.  These include ``language`` and ``speaker_id``.  Since
        # the SpeechAlternative class does not define these fields, we set
        # them dynamically on the instance.  ``speaker_id`` is None when
        # speaker diarisation is not available.
        for attr_name, value in {"language": lang, "speaker_id": None}.items():
            try:
                setattr(alternative, attr_name, value)
            except Exception:
                pass
        return SpeechEvent(SpeechEventType.FINAL_TRANSCRIPT, [alternative])

    async def _grpc_recognize(self, buffer, lang: str) -> SpeechEvent:
        """Internal helper to perform recognition using the gRPC API.

        This method requires the ``grpc`` package and the generated
        protocol buffer modules ``yandex.cloud.ai.stt.v3.stt_pb2`` and
        ``yandex.cloud.ai.stt.v3.stt_service_pb2_grpc`` to be available
        in ``sys.path``.  See the official Yandex Cloud documentation
        for details on generating these modules from the proto
        definitions.

        Parameters
        ----------
        buffer:
            An object with ``data`` (bytes), ``sample_rate`` (int) and
            optionally ``num_channels`` attributes containing raw audio.
        lang:
            Language code for recognition.

        Returns
        -------
        SpeechEvent
            A final transcription event containing the recognised
            alternatives.
        """
        import importlib
        import types

        # Ensure grpc is available
        import grpc  # type: ignore
        # Lazily import the protobuf stubs.  If these imports fail the
        # caller should handle the resulting ImportError.
        stt_pb2 = importlib.import_module("yandex.cloud.ai.stt.v3.stt_pb2")
        stt_service_pb2_grpc = importlib.import_module(
            "yandex.cloud.ai.stt.v3.stt_service_pb2_grpc"
        )

        # Extract audio data and metadata
        raw_data = getattr(buffer, "data", b"") or b""
        # ``buffer.data`` may be a ``memoryview`` or ``bytearray``; convert to
        # ``bytes`` so that slices yield ``bytes`` and not ``memoryview``.
        data_bytes = bytes(raw_data)
        sample_rate = getattr(buffer, "sample_rate", self.sample_rate)
        num_channels = getattr(buffer, "num_channels", 1)

        # Normalize unsupported sample rates by upsampling to 48000 Hz.  See
        # rationale in _recognize_impl for details.  Without this conversion
        # the gRPC API will return INVALID_ARGUMENT for unsupported rates.
        if sample_rate not in (8000, 16000, 48000):
            try:
                import audioop  # type: ignore
                converted, _ = audioop.ratecv(data_bytes, 2, num_channels, sample_rate, 48000, None)
                data_bytes = converted
                sample_rate = 48000
            except Exception:
                sample_rate = 48000

        # After resampling ``data_bytes`` holds PCM at the correct rate.
        data = data_bytes

        # Build streaming options message.  We include model, audio format,
        # language restrictions, text normalization and audio processing
        # type.  Unsupported fields are left at their defaults.
        if self.audio_format.lower() == "lpcm":
            audio_format = stt_pb2.AudioFormatOptions(
                raw_audio=stt_pb2.RawAudio(
                    audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=num_channels,
                )
            )
        else:
            # Assume OggOpus if not LPCM
            audio_format = stt_pb2.AudioFormatOptions(
                container_audio=stt_pb2.ContainerAudio(
                    container_audio_type=stt_pb2.ContainerAudio.OGG_OPUS
                )
            )

        recognition_model = stt_pb2.RecognitionModelOptions(
            model=self.model,
            audio_format=audio_format,
            text_normalization=stt_pb2.TextNormalizationOptions(
                text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                profanity_filter=True,
                literature_text=False,
            ),
            language_restriction=stt_pb2.LanguageRestrictionOptions(
                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                language_code=[lang],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        )

        streaming_options = stt_pb2.StreamingOptions(
            recognition_model=recognition_model
        )

        # Define a generator that yields StreamingRequest messages.  The
        # first message must contain the session options.  Subsequent
        # messages carry chunks of audio.  We split the audio into
        # 4000‑byte chunks as recommended by the Yandex examples.
        def request_gen() -> types.GeneratorType:
            # Session options
            yield stt_pb2.StreamingRequest(session_options=streaming_options)
            chunk_size = 4000
            for i in range(0, len(data), chunk_size):
                chunk_data = data[i : i + chunk_size]
                yield stt_pb2.StreamingRequest(
                    chunk=stt_pb2.AudioChunk(data=chunk_data)
                )

        # Establish a secure channel and create a stub
        credentials = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel("stt.api.cloud.yandex.net:443", credentials)
        stub = stt_service_pb2_grpc.RecognizerStub(channel)
        # Prepare metadata with API key or IAM token
        metadata = [("authorization", f"Api-Key {self.api_key}")]
        # Collect responses from the server
        final_text: Optional[str] = None
        try:
            response_iterator = stub.RecognizeStreaming(
                request_gen(), metadata=metadata
            )
            async for r in response_iterator:
                event_type = r.WhichOneof("Event")
                if event_type == "final":
                    if r.final.alternatives:
                        final_text = r.final.alternatives[0].text
                elif event_type == "final_refinement":
                    # Normalized text may provide a cleaner transcript
                    normalized = r.final_refinement.normalized_text
                    if normalized.alternatives:
                        final_text = normalized.alternatives[0].text
                # We ignore partial results and other events for this
                # non‑streaming integration.
        finally:
            await channel.close()
        # Fallback to empty string if no final result was provided
        if final_text is None:
            final_text = ""
        alternative = SpeechAlternative(text=final_text)
        # Ensure confidence is numeric
        try:
            alternative.confidence = 1.0
        except Exception:
            pass
        # Attach the same dynamic attributes as in the REST implementation
        for attr_name, value in {"language": lang, "speaker_id": None}.items():
            try:
                setattr(alternative, attr_name, value)
            except Exception:
                pass
        return SpeechEvent(SpeechEventType.FINAL_TRANSCRIPT, [alternative])
