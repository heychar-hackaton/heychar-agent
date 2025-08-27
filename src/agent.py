from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, ConversationItemAddedEvent
from livekit.plugins import (
    yandex,
    noise_cancellation,
    silero,
    openai
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Ты HR агент, проводишь собеседование кандидата на должность реакт разработчика. Все английские слова произносятся на русском языке в русской транскрипции.")


async def entrypoint(ctx: agents.JobContext):
    
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    model=f"gpt://{folder_id}/yandexgpt/latest"
    
    session = AgentSession(
        stt=yandex.STT(language="ru-RU"),
        llm=openai.LLM(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=model,
        ),
        tts=yandex.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Поприветствуй пользователя"
    )    


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))