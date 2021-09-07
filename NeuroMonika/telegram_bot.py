import os
import logging
import urllib
import pandas as pd

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.markdown import link
from start_text import *
from functions import *
# from keyboards import *
# import keyboards as kb

# Make sure that u got telegram api token from @BotFather
load_dotenv() #загрузка файла с токеном
TOKEN = os.getenv('TELEGRAM_TOKEN') #подгрузка токена


# Configure logging
#логирование истории обращений пользователя
logging.basicConfig(level=logging.INFO)


# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


# записываем словать руских слов в переменную
dict = pd.read_csv('/NeuroMonika/dicts/rus_dict_50.csv', sep=';')
dict_fonems = pd.DataFrame(pd.read_csv('/NeuroMonika/dicts/fonem_parsing.csv', sep=','))
dict_fonems.mnemonic_list = dict_fonems.mnemonic_list.apply(lambda x : x[1:-1])
dict_fonems.mnemonic_list = dict_fonems.mnemonic_list.str.replace("'", "").str.split(', ')

# Base command messages for start and exceptions (not target content inputs)
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name #сбор персональных данных
    user_id = message.from_user.id #сбор персональных данных
    text = HELLO_TEXT %user_name
    logging.info(f'first start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)


# Проверяем входящие данные на соответствие типу "text"
@dp.message_handler(content_types=['text'])
async def get_word_from_user(message: types.Message):
    user_id = message.from_user.id
    if message.text == 'FAQ':
        descrption = DESCRIPTION
        await message.reply(descrption)
        await bot.send_message(message.from_user.id, 'Надеюсь, так стало понятнее. Введи слово, которое нужно запомнить!')

    elif message.text == 'Начать сначала':
        user_name = message.from_user.first_name  # сбор персональных данных
        text = HELLO_TEXT % user_name
        await bot.send_message(message.from_user.id, text)

    else:
        chat_id = message.chat.id
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        message_id = message.message_id

        text = WAITING_TEXT

        await bot.send_message(chat_id, text)
        p = open("/Users/antonzagorelskii/Desktop/Elbrus/Projects/Google API/image/Monika.gif", 'rb')
        await bot.send_animation(chat_id, p)

        input_text = message.text
        logging.info(f'{user_name, user_id} send this text:{input_text}')
        output = list(translait(input_text))
        output_ru = output[0]
        output_en = output[1]
        output_ruen = output[2]
        output_monika_dict = get_rus_word(output_en, output_ruen)[0]
        output_monika_parsing = get_rus_word(output_en, output_ruen)[1]
        output_photo_mem = get_mem(output_monika_dict, input_text)
        output_respell = respell(output_en)

        if not output_monika_parsing == 'Пока никто не придумал':
            output_text = f'{output_ru}?'\
                          f'\n🧐Перевод "{output_en}"' \
                          f'\n🤪Рунглиш "{output_ruen}"'\
                          f'\n🕵️‍♀️Ассоциативное словосочетание => "{output_monika_parsing}".' \
                          f'\n🧙‍♀️"{output_ruen}" созвучно с "{output_monika_dict}".' \
                          f'\n🤯Фото для закрепления =>'
        else:
            output_text = f'{output_ru}?'\
                          f'\n🧐Перевод "{output_en}".' \
                          f'\n🤪Рунглишь "{output_ruen}"'\
                          f'\n🧙‍♀️"{output_ruen}" созвучно с "{output_monika_dict}".' \
                          f'\n🤯Фото для закрепления =>'

        output_ru = ''
        output_en = ''
        output_respell = ''
        output_ruen = ''
        output_monika_parsing = ''
        output_monika_dict = ''

        await bot.send_message(chat_id, text=output_text)
        if output_photo_mem == 'Фото не нашлось':
            await bot.send_message(chat_id, text="Не нашла")
        else:
            await bot.send_photo(message.chat.id, output_photo_mem)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
