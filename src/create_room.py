import os
import json
import time
import asyncio
import dotenv
import livekit.api

dotenv.load_dotenv(".env")


async def create_interview_room() -> str | None:
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")
    phone_number = os.getenv("CALL_NUMBER")  # или прокиньте аргументом функции
    caller_id = os.getenv("CALL_NUMBER")

    if not url or not api_key or not api_secret:
        print("Нужны LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET в .env")
        return None
    if not sip_trunk_id:
        print("Нужен LIVEKIT_SIP_TRUNK_ID в .env")
        return None
    if not phone_number:
        print("Нужен номер телефона (CALL_NUMBER в .env) в формате E.164, например +7XXXXXXXXXX")
        return None

    interview_metadata = {
        "Должность": "React разработчик",
        "Уровень": "Senior",
        "Компания": "Тук тук Каминоломн",
        "Требования": "React, TypeScript, JavaScript, HTML, CSS",
        "Тип_собеседования": "Техническое интервью",
        "Продолжительность": "45 минут",
        "Язык": "Русский",
    }

    async with livekit.api.LiveKitAPI(url, api_key, api_secret) as lk:
        room_name = f"interview-{int(time.time())}"
        room = await lk.room.create_room(
            livekit.api.CreateRoomRequest(
                name=room_name,
                max_participants=2,
                empty_timeout=120,
                metadata=json.dumps(interview_metadata, ensure_ascii=False),
            )
        )
        print("Комната создана:", room.name, "SID:", room.sid)

        # (опционально) токен для просмотра/отладки в веб-клиенте
        at = livekit.api.AccessToken(api_key, api_secret)
        at.identity = f"user-{int(time.time())}"
        user_token = at.to_jwt()
        print("\nТокен пользователя (скопируйте в клиент):")
        print(user_token)

        # КЛЮЧЕВОЕ: создаём SIP-участника = исходящий звонок через ваш Outbound Trunk
        try:
            req = livekit.api.CreateSIPParticipantRequest(
                room_name=room.name,
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                participant_identity=phone_number,
                wait_until_answered=True,
                # ключевой момент для многих операторов:
                sip_number=caller_id if caller_id else None,
            )
            sip_participant = await lk.sip.create_sip_participant(req)
            print("SIP-участник создан. Статус:", sip_participant.status)
        except livekit.api.TwirpError as e:
            print("Ошибка создания SIP-участника:", e.message)
            # максимально вытащим подсказки из метадаты
            try:
                print("twirp meta:", dict(e.metadata))
            except Exception:
                pass
            return None

        return room.name

if __name__ == "__main__":
    asyncio.run(create_interview_room())
