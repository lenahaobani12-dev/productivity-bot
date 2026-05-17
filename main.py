import os
import re
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger("agentic_bot")

# ─────────────────────────── State ──────────────────────────────
user_tasks: dict[int, list[str]] = defaultdict(list)
# Stores last 5 (role, text) pairs per user
conversation_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))

# ─────────────────────────── Intents ────────────────────────────
INTENT_ADD_TASK       = "add_task"
INTENT_VIEW_TASKS     = "view_tasks"
INTENT_SCHEDULE       = "schedule"
INTENT_TIPS           = "tips"
INTENT_HELP           = "help"
INTENT_GENERAL        = "general_question"
INTENT_UNCLEAR        = "unclear"

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (INTENT_ADD_TASK,   ["أضف", "اضف", "سجل", "احفظ", "ضع", "اريد اضافة", "أريد إضافة", "مهمة جديدة", "add"]),
    (INTENT_VIEW_TASKS, ["مهامي", "مهام", "قائمة", "اعرض", "أعرض", "شوف", "اشوف", "ما عندي", "tasks"]),
    (INTENT_SCHEDULE,   ["جدول", "خطة", "برنامج", "توزيع", "أسبوع", "schedule", "وقت الدراسة"]),
    (INTENT_TIPS,       ["نصائح", "نصيحة", "tips", "اقتراحات", "كيف أتفوق", "كيف أدرس"]),
    (INTENT_HELP,       ["مساعدة", "help", "أوامر", "ماذا تفعل", "ما هي الأوامر", "كيف تعمل"]),
]

PRODUCTIVITY_KEYWORDS = [
    "دراسة", "مذاكرة", "امتحان", "اختبار", "واجب", "مشروع", "تقرير",
    "جامعة", "كلية", "مادة", "محاضرة", "أستاذ", "بحث", "تركيز", "وقت",
    "انتاجية", "إنتاجية", "تنظيم", "قلق", "ضغط", "تعب", "مراجعة", "حفظ",
    "فهم", "تعلم", "مهارة", "ملخص", "نقاط",
]

SYSTEM_PROMPT = (
    "أنت مساعد إنتاجية ذكي ناطق بالعربية للطلاب الجامعيين. "
    "قم دائماً بتحديد نية المستخدم أولاً، ثم اختر سير العمل المناسب. "
    "أجب بشكل احترافي باللغة العربية. لا تختلق معلومات. "
    "إذا كان الطلب غير واضح، اطلب التوضيح."
)

OFF_TOPIC_RESPONSE = (
    "أنا مساعد إنتاجية متخصص للطلاب. "
    "يمكنني مساعدتك في إدارة مهامك وجدولك الدراسي."
)


def detect_intent(text: str) -> str:
    """Classify user message into one of the defined intents."""
    normalized = text.strip().lower()

    for intent, keywords in INTENT_PATTERNS:
        if any(kw in normalized for kw in keywords):
            logger.info("🔍 Intent detected: %s (keyword match)", intent)
            return intent

    # Short unclear messages
    if len(normalized) < 4:
        logger.info("🔍 Intent detected: %s (too short)", INTENT_UNCLEAR)
        return INTENT_UNCLEAR

    # Check if it's productivity-related for general_question
    if any(kw in normalized for kw in PRODUCTIVITY_KEYWORDS):
        logger.info("🔍 Intent detected: %s (productivity keyword)", INTENT_GENERAL)
        return INTENT_GENERAL

    # More than 10 chars with question marks → general question
    if len(normalized) > 10 and ("?" in normalized or "؟" in normalized):
        logger.info("🔍 Intent detected: %s (question detected)", INTENT_GENERAL)
        return INTENT_GENERAL

    if len(normalized) > 20:
        logger.info("🔍 Intent detected: %s (long message, assuming general)", INTENT_GENERAL)
        return INTENT_GENERAL

    logger.info("🔍 Intent detected: %s", INTENT_UNCLEAR)
    return INTENT_UNCLEAR


def get_gemini_client() -> genai.Client:
    replit_base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", "").strip()
    replit_api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "dummy").strip()
    if replit_base_url:
        return genai.Client(
            api_key=replit_api_key,
            http_options={"base_url": replit_base_url},
        )
    google_key = "".join(
        ch for ch in os.environ.get("GOOGLE_API_KEY", "")
        if ch.isprintable() and not ch.isspace()
    )
    return genai.Client(api_key=google_key)


def ask_ai(user_id: int, user_message: str) -> str | None:
    """Call Gemini with conversation history and system prompt."""
    history = list(conversation_history[user_id])

    contents = []
    for role, text in history:
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=600,
                temperature=0.5,
            ),
        )
        reply = (response.text or "").strip()
        if not reply:
            return None
        return reply
    except Exception as e:
        logger.warning("⚠️  Gemini call failed: %s", e)
        return None


def validate_response(text: str | None) -> bool:
    """Validate that a response is non-empty and meaningful."""
    if not text or len(text.strip()) < 2:
        return False
    return True


def record_turn(user_id: int, user_text: str, bot_text: str):
    """Append user + model messages to per-user conversation history."""
    conversation_history[user_id].append(("user", user_text))
    conversation_history[user_id].append(("model", bot_text))


# ─────────────────── Workflow Handlers ──────────────────────────

def workflow_add_task(user_id: int, text: str) -> str:
    """Extract and save a task from natural language."""
    # Strip common prefixes like "أضف مهمة", "اضف", etc.
    task = re.sub(
        r"^(أضف|اضف|سجل|احفظ|ضع|add)\s*(مهمة|task)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()

    if not task:
        return "📌 يرجى كتابة اسم المهمة بعد الأمر.\nمثال: أضف مراجعة الفصل الثالث"

    user_tasks[user_id].append(task)
    count = len(user_tasks[user_id])
    logger.info("✅ Task added for user %d: %s (total: %d)", user_id, task, count)
    return f"✅ تمت إضافة المهمة بنجاح!\n\n📌 {task}\n\nإجمالي مهامك: {count} مهمة"


def workflow_view_tasks(user_id: int) -> str:
    tasks = user_tasks[user_id]
    if not tasks:
        return (
            "📋 قائمة مهامك فارغة حالياً.\n\n"
            "أضف مهمة بكتابة: أضف [اسم المهمة]"
        )
    lines = ["📋 *قائمة مهامك:*\n"]
    for i, task in enumerate(tasks, 1):
        lines.append(f"{i}. {task}")
    lines.append(f"\n*إجمالي: {len(tasks)} مهام*")
    return "\n".join(lines)


DAYS_AR = ["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس"]


def workflow_schedule(user_id: int) -> str:
    tasks = user_tasks[user_id]
    today_idx = datetime.now().weekday()
    day_order = [(today_idx + i) % 7 for i in range(7)]
    day_names = [["السبت", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"][d] for d in day_order]
    work_days = [d for d in day_names if d != "الجمعة"]

    lines = ["📅 *جدولك الدراسي الأسبوعي*\n"]

    if tasks:
        chunks = [tasks[i:i + 2] for i in range(0, len(tasks), 2)]
        for i, day in enumerate(work_days[:5]):
            lines.append(f"🗓 *{day}*")
            if i < len(chunks):
                for task in chunks[i]:
                    lines.append(f"   • {task}  (9:00–11:00 ص)")
                lines.append("   ☕ استراحة 20 دقيقة")
                lines.append("   📖 مراجعة عامة  (4:00–6:00 م)")
            else:
                lines.append("   📖 مراجعة المواد السابقة")
            lines.append("")
    else:
        for day in work_days[:5]:
            lines.append(f"🗓 *{day}*")
            lines.append("   📚 دراسة  (9:00–11:00 ص)")
            lines.append("   ☕ استراحة")
            lines.append("   📖 مراجعة  (4:00–6:00 م)")
            lines.append("")

    lines.append("🗓 *الجمعة*\n   🌟 يوم راحة — استعد للأسبوع القادم!")
    lines.append("\n💡 _أضف مهامك لتخصيص الجدول_")
    return "\n".join(lines)


TIPS_TEXT = (
    "💡 *5 نصائح ذهبية للإنتاجية والدراسة*\n\n"
    "1️⃣ *تقنية بومودورو* 🍅\n"
    "   25 دقيقة تركيز + 5 دقائق راحة. كررها 4 مرات ثم استرح 30 دقيقة.\n\n"
    "2️⃣ *رتب أولوياتك* 📌\n"
    "   ابدأ بالمهمة الأصعب صباحاً وأنت في أوج نشاطك.\n\n"
    "3️⃣ *أوقف الإشعارات* 📵\n"
    "   خصص مكاناً هادئاً وأوقف كل مشتتات الانتباه.\n\n"
    "4️⃣ *راجع كل مساء* 🔄\n"
    "   10 دقائق مراجعة يومية تثبت المعلومات في الذاكرة طويلة المدى.\n\n"
    "5️⃣ *اعتنِ بصحتك* 💪\n"
    "   نوم 7-8 ساعات + تغذية جيدة + نشاط بدني = دماغ أكثر كفاءة."
)

HELP_TEXT = (
    "📖 *دليل الاستخدام*\n\n"
    "📌 `/add [مهمة]` — أضف مهمة\n"
    "   أو اكتب: *أضف مراجعة الرياضيات*\n\n"
    "📋 `/tasks` — اعرض مهامك\n\n"
    "📅 `/schedule` — جدول دراسي أسبوعي\n\n"
    "💡 `/tips` — 5 نصائح للإنتاجية\n\n"
    "💬 *رسالة حرة* — اكتب أي سؤال وسأكشف نيتك وأرد عليك!\n\n"
    "🚀 ابدأ الآن بكتابة مهامك!"
)


async def execute_workflow(
    intent: str,
    user_id: int,
    text: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """Route to the correct workflow based on detected intent."""
    logger.info("⚙️  Executing workflow: %s for user %d", intent, user_id)

    if intent == INTENT_ADD_TASK:
        reply = workflow_add_task(user_id, text)
        await update.message.reply_text(reply, parse_mode="Markdown")
        record_turn(user_id, text, reply)

    elif intent == INTENT_VIEW_TASKS:
        reply = workflow_view_tasks(user_id)
        await update.message.reply_text(reply, parse_mode="Markdown")
        record_turn(user_id, text, reply)

    elif intent == INTENT_SCHEDULE:
        reply = workflow_schedule(user_id)
        await update.message.reply_text(reply, parse_mode="Markdown")
        record_turn(user_id, text, reply)

    elif intent == INTENT_TIPS:
        await update.message.reply_text(TIPS_TEXT, parse_mode="Markdown")
        record_turn(user_id, text, TIPS_TEXT)

    elif intent == INTENT_HELP:
        await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")
        record_turn(user_id, text, HELP_TEXT)

    elif intent == INTENT_GENERAL:
        thinking = await update.message.reply_text("⏳ جاري التفكير...")
        ai_reply = ask_ai(user_id, text)
        if validate_response(ai_reply):
            reply = ai_reply
        else:
            reply = _keyword_fallback(text)
        await thinking.edit_text(reply, parse_mode="Markdown")
        record_turn(user_id, text, reply)

    else:  # INTENT_UNCLEAR
        reply = (
            "لم أفهم طلبك تماماً 🤔\n\n"
            "هل تريد:\n"
            "• إضافة مهمة؟ اكتب: *أضف [المهمة]*\n"
            "• عرض مهامك؟ اكتب: *مهامي*\n"
            "• جدول دراسي؟ اكتب: *جدول*\n"
            "• نصائح؟ اكتب: *نصائح*"
        )
        await update.message.reply_text(reply, parse_mode="Markdown")
        record_turn(user_id, text, reply)


def _keyword_fallback(text: str) -> str:
    """Simple keyword-based fallback when AI is unavailable."""
    t = text.lower()
    if any(k in t for k in ["دراسة", "مذاكرة", "حفظ", "فهم"]):
        return (
            "📚 *نصيحة للدراسة*\n\n"
            "استخدم تقنية بومودورو: 25 دقيقة دراسة + 5 دقائق راحة.\n"
            "راجع ملاحظاتك كل مساء لتثبيت المعلومات. 💪"
        )
    if any(k in t for k in ["امتحان", "اختبار", "مراجعة"]):
        return (
            "📝 *للتحضير للامتحانات*\n\n"
            "• ابدأ المراجعة قبل أسبوع على الأقل\n"
            "• حل أسئلة الامتحانات السابقة\n"
            "• نم جيداً ليلة الامتحان — النوم يثبت المعلومات 🌙"
        )
    if any(k in t for k in ["تعب", "قلق", "ضغط", "توتر"]):
        return (
            "🌟 *للتعامل مع الضغط*\n\n"
            "• خذ استراحة وتنفس بعمق\n"
            "• قسّم المهام إلى خطوات صغيرة\n"
            "• تذكر: كل تقدم صغير هو إنجاز! 💪"
        )
    if not any(k in t for k in PRODUCTIVITY_KEYWORDS):
        return OFF_TOPIC_RESPONSE
    return (
        "🤖 يمكنني مساعدتك في:\n"
        "• إدارة مهامك الدراسية\n"
        "• إنشاء جدول أسبوعي\n"
        "• نصائح الإنتاجية\n\n"
        "اكتب /help لرؤية جميع الأوامر."
    )


# ─────────────────── Command Handlers ───────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "طالب"
    reply = (
        f"مرحباً {name}! 👋\n\n"
        "أنا مساعدك الذكي للإنتاجية الجامعية 🤖\n"
        "أستطيع فهم رسائلك بشكل طبيعي وتحديد ما تحتاجه!\n\n"
        "📌 *ما يمكنني فعله:*\n"
        "• إضافة وعرض مهامك\n"
        "• إنشاء جدول دراسي أسبوعي\n"
        "• الإجابة على أسئلتك الدراسية\n"
        "• نصائح إنتاجية مجربة\n\n"
        "اكتب /help لرؤية الأوامر، أو فقط ابدأ بالكتابة! ✍️"
    )
    await update.message.reply_text(reply, parse_mode="Markdown")
    record_turn(update.effective_user.id, "/start", reply)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "📌 اكتب اسم المهمة بعد الأمر.\nمثال: `/add مراجعة الفصل الثالث`",
            parse_mode="Markdown",
        )
        return
    task = " ".join(context.args)
    reply = workflow_add_task.__wrapped__(user_id, task) if hasattr(workflow_add_task, "__wrapped__") else _direct_add(user_id, task)
    await update.message.reply_text(reply, parse_mode="Markdown")
    record_turn(user_id, f"/add {task}", reply)


def _direct_add(user_id: int, task: str) -> str:
    user_tasks[user_id].append(task)
    count = len(user_tasks[user_id])
    logger.info("✅ /add task for user %d: %s (total: %d)", user_id, task, count)
    return f"✅ تمت إضافة المهمة!\n\n📌 *{task}*\n\nإجمالي مهامك: {count} مهمة"


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reply = workflow_view_tasks(user_id)
    await update.message.reply_text(reply, parse_mode="Markdown")
    record_turn(user_id, "/tasks", reply)


async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reply = workflow_schedule(user_id)
    await update.message.reply_text(reply, parse_mode="Markdown")
    record_turn(user_id, "/schedule", reply)


async def cmd_tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text(TIPS_TEXT, parse_mode="Markdown")
    record_turn(user_id, "/tips", TIPS_TEXT)


async def free_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()

    logger.info("📨 Message from user %d: %s", user_id, text[:80])

    intent = detect_intent(text)
    await execute_workflow(intent, user_id, text, update, context)


# ─────────────────── Keep-Alive Server ──────────────────────────

class _KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("البوت يعمل! ✅".encode())

    def log_message(self, format, *args):
        pass


def start_keep_alive():
    port = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", port), _KeepAliveHandler)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("🌐 Keep-alive server on port %d", port)


# ─────────────────── Entry Point ────────────────────────────────

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    token = "".join(ch for ch in token if ch.isprintable() and not ch.isspace())
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    start_keep_alive()

    request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)
    app = Application.builder().token(token).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("schedule", cmd_schedule))
    app.add_handler(CommandHandler("tips", cmd_tips))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text))

    logger.info("🤖 Agentic bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
