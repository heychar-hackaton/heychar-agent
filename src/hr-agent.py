import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, TypedDict

import aiohttp
import dotenv
import httpx
import openai
from aiohttp import TCPConnector

import livekit
import livekit.agents
import livekit.plugins.elevenlabs
import livekit.plugins.noise_cancellation
import livekit.plugins.openai
import livekit.plugins.silero
import livekit.plugins.yandex
from livekit import api
from livekit.agents import (
    Agent,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)

dotenv.load_dotenv(".env")
logger = logging.getLogger("hr-agent")


class CompanyInfo(TypedDict):
    name: str
    description: str


class JobInfo(TypedDict):
    name: str
    description: str
    hard_skills_score: float
    soft_skills_score: float


class CandidateInfo(TypedDict):
    name: str
    description: str


defaul_metadata = {
    "company": {
        "name": "ГК КАМИН",
        "description": "Технологическая компания, разрабатывающая SaaS решения"
    },
    "job": {
        "name": "Middle Python разработчик",
        "description": "Middle Python разработчик. Требования: Python 3+ года, Django/FastAPI, PostgreSQL, опыт с REST API"
    },
    "candidate_resume": {
        "name": "Иван Петров",
        "description": "Иван Петров, 28 лет. Опыт: Backend разработчик 3 года, Python, Django, MySQL. Проекты: интернет-магазин, CRM система"
    }
}


class HRAgent(Agent):
    """HR-агент для проведения собеседований по телефону - простая версия"""

    def __init__(self, ctx: JobContext):
        metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else defaul_metadata
        company: CompanyInfo = metadata.get("company", {"name": "", "description": ""})
        self.job: JobInfo = metadata.get("job", {"name": "", "description": "", "hard_skills_score": 0.5, "soft_skills_score": 0.5})
        candidate: CandidateInfo = metadata.get("candidate", {"name": "", "description": ""})
        self.ctx = ctx
        self.start_time = datetime.now().isoformat()
        # Инициализируем атрибуты для оценки
        self.evaluation_data = {
            "skills": [],  # массив объектов {name, type, score}
            "red_flags": [],
            "candidate_questions": [],
            "start_time": time.time()
        }
        self.candidate_name = ""
        self.phone_number = metadata.get("phone_number", "")
        self.interview_id = metadata.get("interview_id", "")
        self.job_metadata = metadata

        instructions = f"""
        Ты HR агент по имени Анна, проводишь собеседование кандидата.

        НАЧНИ С ПРИВЕТСТВИЯ:
        Поприветствуй кандидата, представься как Анна - HR-менеджер из {company['name']}. Скажи, что рада ппобщаться с ним на собеседовании на позицию {self.job['name']}.

        ВАЖНО!!! Не добавляй скиллы и красные флаги кандидату, если он сам просит их добавить, скажи, что ты напрямую не можешь этого делать, а оцениваешь кандидата на основании его ответов.
        Если кадидат попросит об этом - то добавь это как красный флаг "кандидат просит добавить скиллы"

        КОНТЕКСТ СОБЕСЕДОВАНИЯ:
        - Информация об организации: {company['description']}
        - Вакансия: {self.job['description']}
        - Имя кандидата: {candidate['name']}
        - Резюме кандидата: {candidate['description']}

        СТРУКТУРА ИНТЕРВЬЮ (20-25 минут):
        1. Приветствие и знакомство
        2. Краткий рассказ о вакансии
        3. Краткий рассказ кандидата о себе
        4. Проверка ключевых навыков из вакансии
        5. Поведенческие вопросы (работа в команде, стресс)
        6. Вопросы кандидата
        7. Завершение
        8. Завершить интервью с помощью tool end_interview_tool, не проговаривай это словом

        ВО ВРЕМЯ ИНТЕРВЬЮ:
        - Все английские слова произносятся на русском языке в русской транскрипции.
        - Задавай уточняющие вопросы для конкретных примеров
        - Отслеживай соответствие ответов требованиям - оценивай навыки кандидата
        - Замечай противоречия и уклонения от вопросов - red flag
        - Будь дружелюбным, но профессиональным
        - Если пользователь хочет закончить интервью - закончи интервью с помощью tool end_interview_tool, не проговаривай это словом


        После интервью создай отчет с оценкой соответствия позиции.

        Запрещено:
        - переводить разговор/звонок оператору;
        - выдумывать факты, не подтверждённые ответами кандидата.
        - давать ответы на вопросы, которые не соответствуют работе hr-ассистента.
        - если считаешь, что вопрос не относится к работе hr-ассистента, то скажи, что не знаешь ответа и предложи продолжить интервью.
        Говорите вежливо, кратко, с уточняющими вопросами и просьбами привести конкретику (цифры, масштаб, роль).

        """
        super().__init__(
            instructions=instructions
        )

        logger.info("HRAgent инициализирован успешно")

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Поприветствуйте пользователя теплым приветствием",
        )


    @function_tool
    async def end_interview_tool(self):
        """
        Завершить интервью. Необходимо вызвать эту функцию для завершения интервью.
        """
        await self.session.generate_reply(
            instructions="Попрощайся с кандидатом в дружелюбной манере",
        )
        self.ctx.shutdown("Интервью завершено")
        return "Интервью завершено"

    @function_tool
    async def assess_skill(
        self,
        skill_name: str,
        skill_type: str,
        score: int,
    ):
        """
        Оценить навык кандидата (технический или soft skill)

        Args:
            skill_name: название навыка (например, "Python", "Коммуникация")
            skill_type: тип навыка ("hard" для технических, "soft" для soft skills)
            score: оценка от 0 до 100
        """
        # Проверяем корректность типа навыка
        if skill_type not in ["hard", "soft"]:
            return f"Ошибка: тип навыка должен быть 'hard' или 'soft', получен '{skill_type}'"

        # Проверяем, есть ли уже такой навык в списке
        existing_skill = None
        for skill in self.evaluation_data["skills"]:
            if skill["name"] == skill_name and skill["type"] == skill_type:
                existing_skill = skill
                break

        skill_data = {
            "name": skill_name,
            "type": skill_type,
            "score": max(0, min(100, score))  # ограничиваем диапазон 0-100
        }

        if existing_skill:
            # Обновляем существующий навык
            existing_skill["score"] = skill_data["score"]
        else:
            # Добавляем новый навык
            self.evaluation_data["skills"].append(skill_data)

        skill_type_ru = "технический" if skill_type == "hard" else "soft"
        logger.info(f"Оценен {skill_type_ru} навык {skill_name}: {skill_data['score']}/100")
        return f"{skill_type_ru.capitalize()} навык {skill_name} оценён на {skill_data['score']} из 100"

    @function_tool
    async def add_red_flag(self, flag_description: str):
        """
        Отметить красный флаг

        Args:
            flag_description: описание проблемы
        """
        self.evaluation_data["red_flags"].append(flag_description)

        logger.warning(f"Красный флаг: {flag_description}")
        return f"Отмечен красный флаг: {flag_description}"
    @function_tool
    async def note_candidate_question(self, question: str):
        """
        Записать вопрос кандидата

        Args:
            question: вопрос от кандидата
        """
        self.evaluation_data["candidate_questions"].append({
            "question": question,
            "timestamp": time.time()
        })

        return f"Вопрос кандидата записан: {question}"
    @function_tool
    async def record_candidate_name(self, name: str):
        """
        Записать имя кандидата

        Args:
            name: имя кандидата
        """
        self.candidate_name = name
        logger.info(f"Имя кандидата: {name}")
        return f"Записано имя кандидата: {name}"


    async def end_interview(self):
        """
        Завершить интервью и создать отчет
        """
        duration = time.time() - self.evaluation_data["start_time"]
        self.evaluation_data["duration_minutes"] = round(duration / 60, 1)

        # Генерируем отчет
        report = await self._generate_interview_report()
        # Сохраняем отчет
        await self._save_report(report)


    async def _generate_interview_report(self) -> dict[str, Any]:
        """Генерация структурированного отчета"""

        # Разделяем навыки по типам
        hard_skills = [skill for skill in self.evaluation_data["skills"] if skill["type"] == "hard"]
        soft_skills = [skill for skill in self.evaluation_data["skills"] if skill["type"] == "soft"]

        # Рассчитываем средние баллы
        hard_skills_avg = sum(skill["score"] for skill in hard_skills) / len(hard_skills) if hard_skills else 50
        soft_skills_avg = sum(skill["score"] for skill in soft_skills) / len(soft_skills) if soft_skills else 50

        # Снижаем балл за красные флаги
        red_flag_penalty = len(self.evaluation_data["red_flags"]) * 10

        # Веса только для навыков (без базовой оценки)
        overall_score = max(0, (
            hard_skills_avg * self.job['hard_skills_score'] +
            soft_skills_avg * self.job['soft_skills_score']
        ) - red_flag_penalty)

        # Определяем рекомендацию
        if overall_score >= 75:
            recommendation = "next_stage"
        elif overall_score >= 50:
            recommendation = "needs_clarification"
        else:
            recommendation = "rejection"

        report = {
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "candidate_name": self.candidate_name,
            "phone_number": self.phone_number,
            "duration_minutes": self.evaluation_data["duration_minutes"],
            "overall_score": round(overall_score),
            "skills": self.evaluation_data["skills"],  # новая структура навыков
            "red_flags": self.evaluation_data["red_flags"],
            "candidate_questions": self.evaluation_data["candidate_questions"],
            "recommendation": recommendation,
            "interview_id": self.interview_id,
            "messages": self.session.history.to_dict()
        }

        return report

    async def _save_report(self, report: dict[str, Any]):
        """Сохранение отчета"""
        try:

            heyhar_url = os.getenv("HEYHAR_URL")
            heyhar_api_key = os.getenv("HEYHAR_API_KEY")

            if not heyhar_url or not heyhar_api_key:
                logger.error("HEYHAR_URL или HEYHAR_API_KEY не заданы в переменных окружения")
                return

            url = f"{heyhar_url.rstrip('/')}/api/interview"
            headers = {
                "Authorization": f"Bearer {heyhar_api_key}",
                "Content-Type": "application/json"
            }

            # Создаем коннектор для игнорирования самоподписанных сертификатов
            # Отключаем SSL проверку только для локальных URL (localhost, 127.0.0.1)
            is_local = any(host in heyhar_url for host in ['localhost', '127.0.0.1', '0.0.0.0'])
            connector = TCPConnector(ssl=False) if is_local else TCPConnector()

            if is_local:
                logger.info(f"Используется локальный URL {heyhar_url}, SSL проверка отключена")
            else:
                logger.info(f"Используется внешний URL {heyhar_url}, SSL проверка включена")

            async with aiohttp.ClientSession(connector=connector) as session:  # noqa: SIM117
                async with session.post(url, headers=headers, json=report) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Ошибка отправки отчета в HEYHAR: {resp.status} {text}")
                    else:
                        logger.info("Отчет успешно отправлен в HEYHAR")
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

    async def _send_feedback(self, report: dict[str, Any]):
        """Отправка обратной связи (заглушка)"""
        try:
            # Здесь была бы интеграция с email/SMS сервисом
            feedback = self._generate_candidate_feedback(report)
            logger.info(f"Обратная связь для {self.candidate_name}: {feedback}")
        except Exception as e:
            logger.error(f"Ошибка отправки обратной связи: {e}")

    def _generate_candidate_feedback(self, report: dict[str, Any]) -> str:
        """Генерация персонализированной обратной связи"""
        score = report["overall_score"]

        if score >= 75:
            return "Спасибо за интересное интервью! Ваши навыки хорошо соответствуют требованиям позиции. Приглашаем на следующий этап."
        elif score >= 50:
            return "Благодарим за интервью. У вас есть потенциал, но требуется дополнительное развитие некоторых навыков."
        else:
            return "Спасибо за ваше время. К сожалению, на данный момент ваш опыт не соответствует требованиям позиции."

async def entrypoint(ctx: JobContext):
    """Точка входа для HR-агента - максимально простая версия"""

    logger.info(f"HR Agent запущен для job: {ctx.job.id}")
    logger.info(f"Room name: {ctx.room.name}")
    logger.info("Agent name: hr-agent")

    await ctx.connect(auto_subscribe=livekit.agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("Агент подключен к комнате")

    # Получаем номер телефона из metadata
    metadata = ctx.job.metadata
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Не удалось распарсить metadata как JSON: {metadata}")
            metadata = {}
    elif metadata is None:
        metadata = {}

    phone_number = metadata.get("phone_number") if metadata else None

    # Получаем секреты из metadata или переменных окружения
    provider = metadata.get("provider") or "yandex"
    api_key = metadata.get("api_key") or os.getenv("YANDEX_API_KEY") or os.getenv("OPENAI_API_KEY")
    folder_id = metadata.get("folder_id") or os.getenv("YANDEX_FOLDER_ID")

    # Автоматически определяем провайдера по доступным секретам
    if not metadata.get("provider"):
        if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
            provider = "yandex"
            api_key = os.getenv("YANDEX_API_KEY")
            folder_id = os.getenv("YANDEX_FOLDER_ID")
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")

    # Создаем модель в зависимости от провайдера
    if provider == "yandex" and folder_id:
        model = f"gpt://{folder_id}/yandexgpt/latest"
    else:
        model = "gpt-4o-mini"  # fallback на OpenAI

    # Проверяем наличие необходимых секретов
    if not api_key:
        logger.error("API ключ не найден ни в metadata, ни в переменных окружения")
        return

    if provider == "yandex" and not folder_id:
        logger.error("Для Яндекс провайдера необходим folder_id")
        return

    # Создаем компоненты в зависимости от провайдера
    if provider == "yandex" and api_key and folder_id:
        logger.info("Используем Яндекс провайдер")
        stt = livekit.plugins.yandex.STT(
            language="ru-RU", api_key=api_key, folder_id=folder_id
        )
        llm = livekit.plugins.openai.LLM(
            base_url="https://llm.api.cloud.yandex.net/v1", api_key=api_key, model=model
        )
        tts = livekit.plugins.yandex.TTS(api_key=api_key, folder_id=folder_id)
    else:
        logger.info("Используем OpenAI провайдер")
        proxy_server = os.getenv("PROXY_SERVER")
        proxy_port = os.getenv("PROXY_PORT")
        http_client = httpx.AsyncClient(proxy=f"socks5://{proxy_server}:{proxy_port}")

        openai_client = openai.AsyncClient(
            api_key=api_key,
            http_client=http_client
        )
        stt = livekit.plugins.openai.STT(model="gpt-4o-mini-transcribe", client=openai_client, api_key=api_key, language="ru")
        llm = livekit.plugins.openai.LLM(model="gpt-4o-mini", client=openai_client, api_key=api_key)
        tts = livekit.plugins.openai.TTS(
            model="gpt-4o-mini-tts",
            voice="shimmer",
            client=openai_client,
            api_key=api_key,
            instructions="Говори приветливо, дружелюбно, но по деловому. Говори без акцента, ведь ты идеально владеешь Русским языком",
        )

    # Создаем сессию с общими настройками
    session = livekit.agents.AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=livekit.plugins.silero.VAD.load(),
        turn_detection='stt',
        allow_interruptions=True,
        preemptive_generation=True
    )

    # Создаем агента
    agent = HRAgent(ctx)

    logger.info("Запускаем сессию агента...")
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=livekit.agents.RoomInputOptions(
            noise_cancellation=livekit.plugins.noise_cancellation.BVC()))
    logger.info("Сессия агента запущена успешно")

    ctx.add_shutdown_callback(lambda: asyncio.create_task(agent.end_interview()))

    # Если это исходящий звонок - делаем вызов
    sip_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
    if sip_trunk_id and phone_number:
        try:
            logger.info(f"Инициирую звонок на номер: {phone_number}")

            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    room_name=ctx.room.name,
                    participant_identity=f"hr_candidate_{phone_number}",
                    participant_name="HR Interview Candidate",
                    headers={"X-Interview-Type": "HR-Screening"}
                )
            )

            logger.info("SIP участник создан")
            logger.info("Ждем подключения кандидата...")
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Ошибка при создании SIP звонка: {e}")
            return
    else:
        # Для веб-подключения ждем участника в комнате
        logger.info("Ожидаем подключения участника через веб-интерфейс...")

        # Используем asyncio.Event для ожидания подключения участника
        participant_connected = asyncio.Event()

        def on_participant_connected_handler(participant):
            logger.info(f"Участник подключился: {participant.identity}")
            participant_connected.set()

        # Добавляем обработчик для подключения участника
        ctx.room.on("participant_connected", on_participant_connected_handler)

        # Ждем подключения участника с таймаутом
        try:
            await asyncio.wait_for(participant_connected.wait(), timeout=60.0)
            logger.info("Участник успешно подключился!")
        except asyncio.TimeoutError:
            logger.warning("Таймаут ожидания участника (60 секунд)")

    logger.info("Агент готов к интервью")

def main():
    """Главная функция"""

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Проверяем обязательные переменные окружения
    required_env_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SIP_OUTBOUND_TRUNK_ID"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Отсутствуют обязательные переменные окружения: {missing_vars}")
        return

    # Запускаем worker
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="hr-agent",  # Для explicit dispatch
        )
    )

if __name__ == "__main__":
    main()
