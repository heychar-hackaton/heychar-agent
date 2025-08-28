from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
import numpy as np

from vosk import Model, KaldiRecognizer
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.stt import SpeechEventType, SpeechEvent, SpeechData, STTCapabilities, RecognitionUsage
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr, APIConnectOptions

import logging
logger = logging.getLogger(__name__)


@dataclass
class STTOptions:
    model_path: NotGivenOr[str]
    language: NotGivenOr[str]
    sample_rate: NotGivenOr[int]
    threshold: NotGivenOr[float]


class STT(stt.STT):
    def __init__(
        self,
        *,
        model_path: NotGivenOr[str] = "./models/vosk/vosk-model-ru",
        language: NotGivenOr[str] = "ru",
        sample_rate: NotGivenOr[int] = 16000,
        threshold: NotGivenOr[float] = 0.5,
    ):
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self._config = STTOptions(
            model_path=model_path,
            language=language,
            sample_rate=sample_rate,
            threshold=threshold,
        )
        
        if not self._config.model_path or not os.path.exists(self._config.model_path):
            raise ValueError(f"Vosk model path is invalid or missing: {self._config.model_path}")
        
        # Ensure we have an integer sample rate
        sample_rate_int = 16000
        if isinstance(self._config.sample_rate, int):
            sample_rate_int = self._config.sample_rate
        
        # Initialize Vosk model and recognizer
        logger.info(f"Initializing Vosk model from {self._config.model_path}")
        self.model = Model(
            model_path=self._config.model_path, 
            lang=self._config.language if isinstance(self._config.language, str) else None
        )
        
        logger.info(f"Creating KaldiRecognizer with sample rate {sample_rate_int}")
        self.recognizer = KaldiRecognizer(self.model, sample_rate_int)

    async def aclose(self) -> None:
        # Clean up Vosk resources if needed
        self.recognizer = None
        self.model = None
        await super().aclose()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise NotImplementedError("Vosk STT does not support single frame recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            recognizer=self.recognizer,
            opts=self._config,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        recognizer: KaldiRecognizer,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        sample_rate = opts.sample_rate if isinstance(opts.sample_rate, int) else 16000
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._opts = opts
        self._recognizer = recognizer
        self._request_id = f"vosk-{id(self)}"
        self._current_text = ""
        self._start_time = time.time()
        self._last_interim_time = 0
        self._total_audio_duration = 0.0
        
        # Vosk needs integer data
        logger.info(f"Initialized VoskSpeechStream with sample rate {sample_rate}")

    def _convert_to_vosk_format(self, frame_data) -> bytes:
        """
        Convert audio frame data to the format expected by Vosk (bytes of int16)
        """
        try:
            if isinstance(frame_data, memoryview):
                # Convert memoryview to bytes
                return bytes(frame_data)
            elif isinstance(frame_data, np.ndarray):
                # Ensure numpy array is int16 and convert to bytes
                return frame_data.astype(np.int16).tobytes()
            elif isinstance(frame_data, bytes):
                # Already bytes, return as is
                return frame_data
            else:
                # Try generic conversion to bytes
                logger.warning(f"Unexpected audio frame data type: {type(frame_data)}")
                return bytes(frame_data)
        except Exception as e:
            logger.error(f"Error converting audio data to Vosk format: {e}", exc_info=True)
            # Return empty bytes as fallback to avoid crashing
            return b''

    async def _run(self) -> None:
        """
        Main processing loop for the Vosk speech recognition stream.
        Processes incoming audio frames, recognizes speech, and emits events.
        """
        # Track if we've detected speech start
        speech_started = False
        
        try:
            async for data in self._input_ch:
                # Handle flush sentinel - indicates end of current segment
                if isinstance(data, self._FlushSentinel):
                    logger.debug("Received flush sentinel")
                    # If we had any accumulated speech, emit final transcript
                    if self._current_text:
                        logger.debug(f"Flushing with current text: {self._current_text}")
                        
                        speech_data = SpeechData(
                            language=self._opts.language if isinstance(self._opts.language, str) else "en",
                            text=self._current_text,
                            start_time=self._start_time,
                            end_time=time.time(),
                            confidence=1.0,
                        )
                        
                        # Emit end of speech
                        self._event_ch.send_nowait(
                            SpeechEvent(
                                type=SpeechEventType.END_OF_SPEECH,
                                request_id=self._request_id,
                            )
                        )
                        
                        # Emit final transcript
                        self._event_ch.send_nowait(
                            SpeechEvent(
                                type=SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=self._request_id,
                                alternatives=[speech_data],
                            )
                        )
                        
                        # Reset current text for next segment
                        self._current_text = ""
                        speech_started = False
                    continue
                
                # Process audio frame
                frame = data
                
                # Update total audio duration for metrics
                frame_duration = frame.samples_per_channel / frame.sample_rate
                self._total_audio_duration += frame_duration
                
                # Debug logging to help diagnose frame data type issues
                # logger.debug(f"Processing audio frame: samples={frame.samples_per_channel}, " 
                #            f"data type={type(frame.data)}, "
                #            f"shape={getattr(frame.data, 'shape', None)}")
                
                # Convert frame to bytes in the format Vosk expects
                try:
                    samples = self._convert_to_vosk_format(frame.data)
                    
                    # Skip empty frames
                    if not samples:
                        logger.warning("Empty audio frame detected, skipping")
                        continue
                    
                    # Process the audio data - this is where the original error occurred
                    if self._recognizer.AcceptWaveform(samples):
                        # We have a final result
                        result_json = self._recognizer.Result()
                        # logger.debug(f"Vosk final result: {result_json}")
                        result = json.loads(result_json)
                        
                        if "text" in result and result["text"].strip():
                            text = result["text"].strip()
                            logger.debug(f"Final text: {text}")
                            
                            # If this is our first speech, emit start of speech
                            if not speech_started:
                                speech_started = True
                                self._start_time = time.time()
                                logger.debug("Emitting START_OF_SPEECH event")
                                self._event_ch.send_nowait(
                                    SpeechEvent(
                                        type=SpeechEventType.START_OF_SPEECH,
                                        request_id=self._request_id,
                                    )
                                )
                            
                            # Update current text
                            self._current_text = text
                            
                            # Create speech data
                            speech_data = SpeechData(
                                language=self._opts.language if isinstance(self._opts.language, str) else "en",
                                text=text,
                                start_time=self._start_time,
                                end_time=time.time(),
                                confidence=1.0,
                            )
                            
                            # Emit final transcript event
                            logger.debug("Emitting FINAL_TRANSCRIPT event")
                            self._event_ch.send_nowait(
                                SpeechEvent(
                                    type=SpeechEventType.FINAL_TRANSCRIPT,
                                    request_id=self._request_id,
                                    alternatives=[speech_data],
                                )
                            )
                            
                            # Emit usage metrics
                            self._event_ch.send_nowait(
                                SpeechEvent(
                                    type=SpeechEventType.RECOGNITION_USAGE,
                                    request_id=self._request_id,
                                    recognition_usage=RecognitionUsage(
                                        audio_duration=self._total_audio_duration
                                    ),
                                )
                            )
                    else:
                        # We have an interim result
                        result_json = self._recognizer.PartialResult()
                        # logger.debug(f"Vosk partial result: {result_json}")
                        result = json.loads(result_json)
                        
                        # Only process if we have partial text and enough time has passed since last interim
                        current_time = time.time()
                        if ("partial" in result and result["partial"].strip() and 
                                (current_time - self._last_interim_time) > 0.1):  # Limit interim updates
                            
                            text = result["partial"].strip()
                            #logger.debug(f"Partial text: {text}")
                            
                            # If this is our first speech, emit start of speech
                            if not speech_started:
                                speech_started = True
                                self._start_time = current_time
                                logger.debug("Emitting START_OF_SPEECH event")
                                self._event_ch.send_nowait(
                                    SpeechEvent(
                                        type=SpeechEventType.START_OF_SPEECH,
                                        request_id=self._request_id,
                                    )
                                )
                            
                            # Create speech data
                            speech_data = SpeechData(
                                language=self._opts.language if isinstance(self._opts.language, str) else "en",
                                text=text,
                                start_time=self._start_time,
                                end_time=current_time,
                                confidence=0.5,  # Default confidence for interim results
                            )
                            
                            # Emit interim transcript event
                           # logger.debug("Emitting INTERIM_TRANSCRIPT event")
                            self._event_ch.send_nowait(
                                SpeechEvent(
                                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                                    request_id=self._request_id,
                                    alternatives=[speech_data],
                                )
                            )
                            
                            self._last_interim_time = current_time
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}", exc_info=True)
                    # Continue processing other frames instead of crashing
                    continue
            
            # Input channel closed, finalize any remaining recognition
            logger.debug("Input channel closed, finalizing recognition")
            if speech_started and self._current_text:
                # Emit end of speech
                logger.debug("Emitting final END_OF_SPEECH event")
                self._event_ch.send_nowait(
                    SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        request_id=self._request_id,
                    )
                )
                
                # Get final result if there's any pending
                try:
                    final_result_json = self._recognizer.FinalResult()
                    logger.debug(f"Vosk final flush result: {final_result_json}")
                    final_result = json.loads(final_result_json)
                    
                    if "text" in final_result and final_result["text"].strip():
                        text = final_result["text"].strip()
                        logger.debug(f"Final flush text: {text}")
                        
                        # Create speech data
                        speech_data = SpeechData(
                            language=self._opts.language if isinstance(self._opts.language, str) else "en",
                            text=text,
                            start_time=self._start_time,
                            end_time=time.time(),
                            confidence=1.0,
                        )
                        
                        # Emit final transcript event
                        logger.debug("Emitting final FINAL_TRANSCRIPT event")
                        self._event_ch.send_nowait(
                            SpeechEvent(
                                type=SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=self._request_id,
                                alternatives=[speech_data],
                            )
                        )
                except Exception as e:
                    logger.error(f"Error getting final result: {e}", exc_info=True)
                
                # Final usage metrics
                self._event_ch.send_nowait(
                    SpeechEvent(
                        type=SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._request_id,
                        recognition_usage=RecognitionUsage(
                            audio_duration=self._total_audio_duration
                        ),
                    )
                )
        
        except Exception as e:
            logger.error(f"Error in Vosk speech stream: {e}", exc_info=True)
            raise

    async def aclose(self) -> None:
        """Close the stream and clean up resources"""
        logger.debug("Closing Vosk speech stream")
        # Reset the recognizer if needed
        try:
            self._recognizer.Reset()
        except Exception as e:
            logger.error(f"Error resetting Vosk recognizer: {e}", exc_info=True)
        
        await super().aclose()