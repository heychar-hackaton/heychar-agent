import json
import dotenv
import os
import livekit
import livekit.agents
import livekit.plugins
import livekit.plugins.openai
import livekit.plugins.yandex
import livekit.plugins.silero
import livekit.plugins.turn_detector
import livekit.plugins.noise_cancellation

dotenv.load_dotenv(".env")


class Assistant(livekit.agents.Agent):
    def __init__(self, *, chat_ctx: livekit.agents.ChatContext | None) -> None:
        super().__init__(
            instructions="Ты HR агент, проводишь собеседование кандидата на должность реакт разработчика. "
                         "Все английские слова произносятся на русском языке в русской транскрипции.",
            chat_ctx=chat_ctx)

async def entrypoint(ctx: livekit.agents.JobContext):
    await ctx.connect(auto_subscribe=livekit.agents.AutoSubscribe.AUDIO_ONLY)

    try:
        room_metadata = json.loads(raw_data) if (raw_data := ctx.room.metadata) else {}
    except json.JSONDecodeError:
        room_metadata = {}

    if metadata_str := '\n'.join([f"{k}: {v}" for k, v in room_metadata.items()]):
        chat_ctx = livekit.agents.ChatContext(
            items=[livekit.agents.ChatMessage(
                role='system',
                content=[metadata_str])])
    else:
        chat_ctx = None

    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    model=f"gpt://{folder_id}/yandexgpt/latest"

    session = livekit.agents.AgentSession(
        stt=livekit.plugins.yandex.STT(language="ru-RU"),
        llm=livekit.plugins.openai.LLM(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=model),
        tts=livekit.plugins.yandex.TTS(),
        vad=livekit.plugins.silero.VAD.load(),
        turn_detection='stt',
        preemptive_generation=True)

    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=chat_ctx),
        room_input_options=livekit.agents.RoomInputOptions(
            noise_cancellation=livekit.plugins.noise_cancellation.BVC()))

    await session.generate_reply(instructions="Поприветствуй пользователя")


if __name__ == "__main__":
    livekit.agents.cli.run_app(livekit.agents.WorkerOptions(entrypoint_fnc=entrypoint))
