from transformers import T5ForConditionalGeneration, T5Tokenizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import nest_asyncio
import asyncio
from langdetect import detect, DetectorFactory

# Исправление проблемы с циклом событий (Event Loop) при использовании asyncio
nest_asyncio.apply()

# Установка фиксированного значения случайности для библиотеки langdetect для обеспечения стабильных результатов
DetectorFactory.seed = 0

# Загрузка модели
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)  # Использование нового поведения токенизатора
model.eval()

# Функция перефразирования
def paraphrase(text, sequences=1, beams=15, grams=4, do_sample=True):
    # Преобразование текста в формат, подходящий для модели
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)  # Определение максимальной длины выходного текста
    # Генерация перефразированных текстов с использованием модели
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams,
                         num_beams=beams, max_length=max_size,
                         num_return_sequences=sequences,
                         do_sample=do_sample)
    # Декодирование сгенерированных текстов и их возврат
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# Функция /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Очистка любых предыдущих данных пользователя
    context.user_data.clear()
    # Установка начального состояния пользователя
    context.user_data['state'] = 'waiting_for_text'
    # Отправка приветственного сообщения пользователю
    await update.message.reply_text(
        "👋 Привет! Я бот для перефразирования текста.\n"
        "Просто отправьте мне текст, и я перефразирую его для вас.\n\n"
        "📚 Команды:\n"
        "/help - Инструкция по использованию"
    )

# Функция /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Отправка инструкции по использованию пользователю
    await update.message.reply_text(
        "📖 Инструкция:\n"
        "1. Отправьте текст, который вы хотите перефразировать.\n"
        "2. Укажите количество вариантов перефразирования (до 15).\n"
        "3. Я верну вам несколько вариантов перефразирования.\n"
        "4. Если результат вас не устраивает, я попробую снова.\n\n"
        "💡 Подсказка: Вы можете отправить текст длиной до 500 символов."
    )

# Обработка сообщений в зависимости от состояния
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получение текущего состояния пользователя
    state = context.user_data.get('state', 'waiting_for_text')
    user_input = update.message.text

    if state == 'waiting_for_text':
        # Проверка длины текста
        if len(user_input) > 500:
            await update.message.reply_text("⚠️ Текст слишком длинный. Пожалуйста, отправьте текст до 500 символов.")
            return

        # Проверка языка текста
        try:
            detected_language = detect(user_input)
            if detected_language != 'ru':  # Если язык не русский
                await update.message.reply_text("⚠️ Пожалуйста, отправьте текст на русском языке. Мы принимаем только русский язык, Пожалуйста, введите еще раз на русском языке,")
                return
        except Exception as e:
            await update.message.reply_text(f"❌ Произошла ошибка при определении языка: {e}")
            return

        # Сохранение текста и переход к запросу количества вариантов
        context.user_data['text_to_paraphrase'] = user_input
        context.user_data['state'] = 'waiting_for_options'
        await update.message.reply_text("🔢 Сколько вариантов перефразирования вы хотите? (до 15)")

    elif state == 'waiting_for_options':
        # Проверка, что введено целое число
        try:
            sequences = int(user_input)
            if sequences < 1 or sequences > 15:
                await update.message.reply_text("⚠️ Пожалуйста, введите число от 1 до 15.")
                return

            # Сохранение количества вариантов и переход к перефразированию
            context.user_data['sequences'] = sequences
            context.user_data['state'] = 'waiting_for_feedback'

            # Перефразирование текста
            await update.message.reply_text("🔄 Обрабатываю текст...")  # Сообщение во время обработки
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            # Отправка результатов
            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"✨ Вот варианты перефразирования:\n{response}")

            # Вопрос пользователю, удовлетворен ли он результатом
            await update.message.reply_text("❓ Вас устраивает результат? (Да/Нет)")
            context.user_data['paraphrased_texts'] = paraphrased_texts  # Сохранение результатов для повторного использования
        except ValueError:
            await update.message.reply_text("⚠️ Пожалуйста, введите число от 1 до 15.")

    elif state == 'waiting_for_feedback':
        feedback = user_input.lower()
        if feedback == "да":
            await update.message.reply_text("🎉 Отлично! Рад, что вам понравилось. Если хотите, отправьте новый текст.")
            context.user_data.clear()  # Очистка временных данных
            context.user_data['state'] = 'waiting_for_text'  # Возврат к состоянию ожидания текста
        elif feedback == "нет":
            await update.message.reply_text("🔄 Попробую снова...")
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            sequences = context.user_data['sequences']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            # Отправка новых результатов
            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"✨ Вот новые варианты перефразирования:\n{response}")
            await update.message.reply_text("❓ Вас устраивает результат? (Да/Нет)")
        else:
            await update.message.reply_text("⚠️ Пожалуйста, ответьте 'Да' или 'Нет'.")

# Настройка бота
async def main():
    # Укажите ваш токен здесь
    TOKEN = "7979216405:AAGWpD07_1D9isAXZoIM2TRO_h7sSHGZbFQ"

    # Настройка приложения
    application = Application.builder().token(TOKEN).build()

    # Добавление команд и обработчиков сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
