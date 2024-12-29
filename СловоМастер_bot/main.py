from transformers import T5ForConditionalGeneration, T5Tokenizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import nest_asyncio
import asyncio
from langdetect import detect, DetectorFactory

# إصلاح مشكلة حلقة الأحداث
nest_asyncio.apply()

# تثبيت عشوائية ثابتة لمكتبة langdetect لضمان نتائج متسقة
DetectorFactory.seed = 0

# تحميل النموذج
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)  # استخدام السلوك الجديد
model.eval()

# وظيفة إعادة الصياغة
def paraphrase(text, sequences=1, beams=15, grams=4, do_sample=True):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams,
                         num_beams=beams, max_length=max_size,
                         num_return_sequences=sequences,
                         do_sample=do_sample)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# وظيفة /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()  # مسح أي بيانات سابقة
    context.user_data['state'] = 'waiting_for_text'  # تحديد الحالة الأولى
    await update.message.reply_text(
        "👋 Привет! Я бот для перефразирования текста.\n"
        "Просто отправьте мне текст, и я перефразирую его для вас.\n\n"
        "📚 Команды:\n"
        "/help - Инструкция по использованию"
    )

# وظيفة /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Инструкция:\n"
        "1. Отправьте текст, который вы хотите перефразировать.\n"
        "2. Укажите количество вариантов перефразирования (до 15).\n"
        "3. Я верну вам несколько вариантов перефразирования.\n"
        "4. Если результат вас не устраивает, я попробую снова.\n\n"
        "💡 Подсказка: Вы можете отправить текст длиной до 500 символов."
    )

# التعامل مع الرسائل بناءً على الحالة
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get('state', 'waiting_for_text')  # الحصول على الحالة الحالية
    user_input = update.message.text

    if state == 'waiting_for_text':
        # التحقق من طول النص
        if len(user_input) > 500:
            await update.message.reply_text("⚠️ Текст слишком длинный. Пожалуйста, отправьте текст до 500 символов.")
            return

        # التحقق من اللغة
        try:
            detected_language = detect(user_input)
            if detected_language != 'ru':  # إذا لم تكن اللغة الروسية
                await update.message.reply_text("⚠️ Пожалуйста, отправьте текст на русском языке. Мы принимаем только русский язык, Пожалуйста, введите еще раз на русском языке,")
                return
        except Exception as e:
            await update.message.reply_text(f"❌ Произошла ошибка при определении языка: {e}")
            return

        # حفظ النص والانتقال إلى حالة طلب عدد الخيارات
        context.user_data['text_to_paraphrase'] = user_input
        context.user_data['state'] = 'waiting_for_options'
        await update.message.reply_text("🔢 Сколько вариантов перефразирования вы хотите? (до 15)")

    elif state == 'waiting_for_options':
        # التحقق من أن المدخل هو رقم صحيح
        try:
            sequences = int(user_input)
            if sequences < 1 or sequences > 15:
                await update.message.reply_text("⚠️ Пожалуйста, введите число от 1 до 15.")
                return

            # حفظ عدد الخيارات والانتقال إلى حالة إعادة الصياغة
            context.user_data['sequences'] = sequences
            context.user_data['state'] = 'waiting_for_feedback'

            # إعادة صياغة النص
            await update.message.reply_text("🔄 Обрабатываю текст...")  # رسالة أثناء المعالجة
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            # إرسال النتائج
            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"✨ Вот варианты перефразирования:\n{response}")

            # سؤال المستخدم إذا كان راضيًا
            await update.message.reply_text("❓ Вас устраивает результат? (Да/Нет)")
            context.user_data['paraphrased_texts'] = paraphrased_texts  # حفظ النتائج لإعادة استخدامها
        except ValueError:
            await update.message.reply_text("⚠️ Пожалуйста, введите число от 1 до 15.")

    elif state == 'waiting_for_feedback':
        feedback = user_input.lower()
        if feedback == "да":
            await update.message.reply_text("🎉 Отлично! Рад, что вам понравилось. Если хотите, отправьте новый текст.")
            context.user_data.clear()  # مسح البيانات المؤقتة
            context.user_data['state'] = 'waiting_for_text'  # العودة إلى حالة انتظار النص
        elif feedback == "нет":
            await update.message.reply_text("🔄 Попробую снова...")
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            sequences = context.user_data['sequences']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            # إرسال النتائج الجديدة
            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"✨ Вот новые варианты перефразирования:\n{response}")
            await update.message.reply_text("❓ Вас устраивает результат? (Да/Нет)")
        else:
            await update.message.reply_text("⚠️ Пожалуйста, ответьте 'Да' или 'Нет'.")

# إعداد البوت
async def main():
    # ضع التوكن الخاص بك هنا
    TOKEN = "7979216405:AAGWpD07_1D9isAXZoIM2TRO_h7sSHGZbFQ"

    # إعداد التطبيق
    application = Application.builder().token(TOKEN).build()

    # إضافة الأوامر ومعالجات الرسائل
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # بدء تشغيل البوت
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())