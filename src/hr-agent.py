import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import dotenv

import livekit
import livekit.agents
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

# Инструкции для HR-агента
HR_INSTRUCTIONS = """
Ты профессиональный HR-специалист, проводишь собеседование на русском языке по телефону.
Все английские слова произноси в русской транскрипции.

КОНТЕКСТ СОБЕСЕДОВАНИЯ:
- Информация об организации: {company_info}
- Вакансия: {job_description}  
- Резюме кандидата: {candidate_resume}

СТРУКТУРА ИНТЕРВЬЮ (20-25 минут):
1. Приветствие и знакомство
2. Краткий рассказ кандидата о себе
3. Проверка ключевых навыков из вакансии
4. Поведенческие вопросы (работа в команде, стресс)
5. Вопросы кандидата
6. Завершение

ВО ВРЕМЯ ИНТЕРВЬЮ:
- Задавай уточняющие вопросы для конкретных примеров
- Отслеживай соответствие ответов требованиям
- Замечай противоречия и уклонения от вопросов
- Будь дружелюбным, но профессиональным

После интервью создай отчет с оценкой соответствия позиции.
"""

class HRAgent(Agent):
    """HR-агент для проведения собеседований по телефону - простая версия"""

    def __init__(self, ctx: JobContext):
        metadata = json.loads(ctx.job.metadata)
        org_text = metadata.get("company_info", "")
        vacancy_text = metadata.get("job_description", "")
        resume_text = metadata.get("candidate_resume", "")

        instructions = f"""
        Ты HR агент, проводишь собеседование кандидата на должность разработчика. 
        Все английские слова произносятся на русском языке в русской транскрипции.
        Будь дружелюбным и профессиональным. Задавай вопросы о опыте работы и навыках.

        Контексты (не озвучивать, использовать для анализа):
        [ОРГАНИЗАЦИЯ]
        {org_text[:2000]}

        [ВАКАНСИЯ]
        {vacancy_text[:4000]}

        [РЕЗЮМЕ]
        {resume_text[:4000]}

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

    def _create_llm(self):
        """Создание LLM с Yandex GPT"""
        api_key = os.getenv("YANDEX_API_KEY")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not api_key or not folder_id:
            raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID должны быть установлены")

        model = f"gpt://{folder_id}/yandexgpt/latest"

        return livekit.plugins.openai.LLM(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=model,
            temperature=0.7
        )

    def _build_context(self, metadata: dict) -> livekit.agents.ChatContext:
        """Построение контекста чата с информацией о вакансии"""
        # Извлекаем данные из metadata
        company_info = metadata.get("company_info", "Информация о компании не предоставлена")
        job_description = metadata.get("job_description", "Описание вакансии не предоставлено")
        candidate_resume = metadata.get("candidate_resume", "Резюме кандидата не предоставлено")

        # Форматируем инструкции с контекстом
        formatted_instructions = HR_INSTRUCTIONS.format(
            company_info=company_info,
            job_description=job_description,
            candidate_resume=candidate_resume
        )

        chat_ctx = livekit.agents.ChatContext(
            items=[livekit.agents.ChatMessage(
                role='system',
                content=[formatted_instructions]
            )]
        )

        return chat_ctx

    @function_tool
    async def assess_technical_skill(
        self,
        skill_name: str,
        claimed_level: str,
        evidence: str,
        assessment: str
    ):
        """
        Оценить технический навык кандидата
        
        Args:
            skill_name: название навыка (например, "Python", "SQL")
            claimed_level: заявленный уровень (Junior/Middle/Senior)
            evidence: доказательства/примеры от кандидата
            assessment: оценка (confirmed/partial/not_confirmed)
        """
        self.evaluation_data["technical_skills"][skill_name] = {
            "claimed_level": claimed_level,
            "evidence": evidence,
            "assessment": assessment,
            "timestamp": time.time()
        }

        logger.info(f"Оценен навык {skill_name}: {assessment}")
        return f"Навык {skill_name} оценён как {assessment}"

    @function_tool
    async def assess_soft_skill(
        self,
        skill_name: str,
        score: int,
        notes: str
    ):
        """
        Оценить soft skill кандидата
        
        Args:
            skill_name: название навыка (коммуникация, работа в команде)
            score: оценка от 1 до 10
            notes: комментарии
        """
        self.evaluation_data["soft_skills"][skill_name] = {
            "score": score,
            "notes": notes,
            "timestamp": time.time()
        }

        logger.info(f"Soft skill {skill_name}: {score}/10")
        return f"Soft skill {skill_name} оценён на {score} из 10"

    @function_tool
    async def add_red_flag(self, flag_description: str):
        """
        Отметить красный флаг
        
        Args:
            flag_description: описание проблемы
        """
        self.evaluation_data["red_flags"].append({
            "description": flag_description,
            "timestamp": time.time()
        })

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

    @function_tool
    async def end_interview(self):
        """
        Завершить интервью и создать отчет
        """
        duration = time.time() - self.evaluation_data["start_time"]
        self.evaluation_data["duration_minutes"] = round(duration / 60, 1)

        # Генерируем отчет
        report = await self._generate_interview_report()

        # Сохраняем отчет (в реальной системе - в БД)
        await self._save_report(report)

        # Планируем отправку обратной связи
        asyncio.create_task(self._send_feedback(report))

        return "Интервью завершено. Спасибо за время! Результаты будут отправлены в течение 3 дней."

    async def _generate_interview_report(self) -> dict[str, Any]:
        """Генерация структурированного отчета"""

        # Рассчитываем общий балл
        technical_scores = []
        for skill_data in self.evaluation_data["technical_skills"].values():
            if skill_data["assessment"] == "confirmed":
                technical_scores.append(100)
            elif skill_data["assessment"] == "partial":
                technical_scores.append(60)
            else:
                technical_scores.append(20)

        soft_skill_scores = [
            skill_data["score"] * 10
            for skill_data in self.evaluation_data["soft_skills"].values()
        ]

        # Веса: технические навыки 50%, soft skills 30%, общая оценка 20%
        technical_avg = sum(technical_scores) / len(technical_scores) if technical_scores else 50
        soft_skills_avg = sum(soft_skill_scores) / len(soft_skill_scores) if soft_skill_scores else 50

        # Снижаем балл за красные флаги
        red_flag_penalty = len(self.evaluation_data["red_flags"]) * 10

        overall_score = max(0, (
            technical_avg * 0.5 +
            soft_skills_avg * 0.3 +
            70 * 0.2  # базовая оценка за проведенное интервью
        ) - red_flag_penalty)

        # Определяем рекомендацию
        if overall_score >= 75:
            recommendation = "next_stage"
        elif overall_score >= 50:
            recommendation = "clarification_needed"
        else:
            recommendation = "reject"

        report = {
            "candidate_name": self.candidate_name,
            "phone_number": self.phone_number,
            "interview_date": datetime.now().isoformat(),
            "duration_minutes": self.evaluation_data["duration_minutes"],
            "overall_score": round(overall_score),
            "technical_assessment": self.evaluation_data["technical_skills"],
            "soft_skills": self.evaluation_data["soft_skills"],
            "red_flags": self.evaluation_data["red_flags"],
            "candidate_questions": self.evaluation_data["candidate_questions"],
            "recommendation": recommendation,
            "job_metadata": self.job_metadata
        }

        return report

    async def _save_report(self, report: dict[str, Any]):
        """Сохранение отчета"""
        try:
            # В реальной системе здесь была бы запись в БД
            filename = f"interview_report_{self.phone_number}_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"Отчет сохранен: {filename}")
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
    logger.info(f"Metadata: {ctx.job.metadata}")

    # Подключаемся к комнате (точно как в agent.py)
    await ctx.connect(auto_subscribe=livekit.agents.AutoSubscribe.AUDIO_ONLY)

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

        except Exception as e:
            logger.error(f"Ошибка при создании SIP звонка: {e}")
            return

    # Создаем простого агента (точно как в agent.py)
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
    model = f"gpt://{folder_id}/yandexgpt/latest"

    session = livekit.agents.AgentSession(
        stt=livekit.plugins.yandex.STT(language="ru-RU"),
        llm=livekit.plugins.openai.LLM(
            base_url="https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=model),
        tts=livekit.plugins.yandex.TTS(),
        vad=livekit.plugins.silero.VAD.load(),
        turn_detection='stt',
        allow_interruptions=True,
        min_interruption_words=3,
        preemptive_generation=True)

    await session.start(
        room=ctx.room,
        agent=HRAgent(ctx),
        room_input_options=livekit.agents.RoomInputOptions(
            noise_cancellation=livekit.plugins.noise_cancellation.BVC()))


    logger.info("Отправляем приветствие")
    await session.generate_reply(
        instructions="Поприветствуй кандидата, если ты не знаешь, как его зовут, то спроси, как можно к нему обращаться.",
        allow_interruptions=False
    )

def main():
    """Главная функция"""

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Проверяем переменные окружения
    required_env_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "YANDEX_API_KEY",
        "YANDEX_FOLDER_ID",
        "SIP_OUTBOUND_TRUNK_ID"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Отсутствуют переменные окружения: {missing_vars}")
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

# Пример использования для создания dispatch:
# lk dispatch create \
#   --new-room \
#   --agent-name hr-interviewer \
#   --metadata '{
#     "phone_number": "+71234567890",
#     "company_info": "Технологическая компания, разрабатывающая SaaS решения",
#     "job_description": "Middle Python разработчик. Требования: Python 3+ года, Django/FastAPI, PostgreSQL, опыт с REST API",
#     "candidate_resume": "Иван Петров, 28 лет. Опыт: Backend разработчик 3 года, Python, Django, MySQL. Проекты: интернет-магазин, CRM система"
#   }'
