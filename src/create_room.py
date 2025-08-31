import asyncio
import json
import os
import time

import dotenv
import livekit.api

dotenv.load_dotenv(".env")


async def create_interview_room() -> str | None:
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not url or not api_key or not api_secret:
        print("Нужны LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET в .env")
        return None

    # Пример «комнатного» контекста (вы потом будете читать его в воркере из room.metadata)
    interview_metadata = {
        "Должность": "React разработчик",
        "Уровень": "Senior",
        "Компания": "Тук тук Каминоломн",
        "Требования": "React, TypeScript, JavaScript, HTML, CSS",
        "Тип_собеседования": "Техническое интервью",
        "Продолжительность": "45 минут",
        "Язык": "Русский",
    }

    # Рекомендуемый путь: LiveKitAPI → lk.room
    async with livekit.api.LiveKitAPI(url, api_key, api_secret) as lk:
        room_name = f"interview-{int(time.time())}"

        room = await lk.room.create_room(
            livekit.api.CreateRoomRequest(
                name=room_name,
                max_participants=2,          # под 1-на-1
                empty_timeout=120,           # авто-закрытие пустой комнаты
                metadata=json.dumps(interview_metadata, ensure_ascii=False),
            )
        )

        print("Комната создана:", room.name, "SID:", room.sid)

        # Выпускаем пользовательский токен для входа в комнату
        at = livekit.api.AccessToken(api_key, api_secret)
        at.identity = f"user-{int(time.time())}"
        user_token = at.to_jwt()

        print("\nТокен пользователя (скопируйте в клиент):")
        print(user_token)

        return room.name


if __name__ == "__main__":
    asyncio.run(create_interview_room())
