import pronouncing
import re
import difflib
import random
import lytspel
import requests

from telegram_bot import dict, dict_fonems
from googletrans import Translator
from dict_fonem import dict_fonem
from ImageParser import YandexImage



word = ''
runglish = ''
# Функция перевод. Возвращает слово на русском, перевод на eng, произношение.
def translait(get_word):
    translator = Translator()  # cоздаем обьект класса Translator

    # Записываем в переменные выхода нули, чтобы они очишались при каждом вызове
    word = 0
    word_rus = 0
    result_en = 0
    cleaned = []  # создаем лист для записи фонем без ударения
    rus_w = []  # создаем лист для записи сопоставленных значений фонем и русского произношения
    runglish = []  # хранит на выходе произношение русскими буквами
    mnenon = 0

    word = get_word.lower()  # переводим полученное слово в нижний регистр
    word_rus = word  # сохраняем входящее слово
    result_en = translator.translate(word).text.lower()  # записываем перевод
    print(result_en)
    if len(result_en.split(' ')) > 1:  # для случаев когда переводчик возвращает перевод с артиклем, отрезаем артикль
        result_en = result_en.split(' ')[1]

    if len(pronouncing.phones_for_word(result_en)) > 0:  # условие нахождения фонемы
        result_eng = ''.join(pronouncing.phones_for_word(result_en)[0]).split(
            ' ')  # получаем фонемы и сплитуем их для дальнейшего перебора
        print(result_eng)

        for i in result_eng:  # перебираем значения в полученных фонемах и удаляем логические ударения
            clean = re.sub(r'\d+$', '', i)  # регулярное выражение для удаления цифр (логических ударений)
            print(clean)
            cleaned.append(clean)  # записываем полученные фонемы без логических ударений в лист

        for i in cleaned:  # сопоставляем фонемы со словарем русского произношения
            ru = dict_fonem[i]  # обращение в словарь по ключу, равному фонеме
            # print(ru)
            rus_w.append(ru)  # добавляем русское произношение фонемы в лист
            # print(rus_w)
        runglish = str(''.join(rus_w).lower())  # собираем русское произношение
    else:
        print("Мы такого слова не знаем")
        runglish = 'nope'

    mnenon = get_rus_word(result_en, runglish)

    return [word_rus, result_en, runglish, mnenon, word_rus]


# Функция поиска слова в рускоязычном словаре по транскрипции Ливенштейна
def get_rus_word(result_eng, word_rus):
    # Поиск по словарю с парсингом
    if result_eng in dict_fonems['english'].values:
        index = dict_fonems.loc[dict_fonems['english'] == result_eng].index[0]
        i = len(dict_fonems.loc[index, 'mnemonic_list'])
        mnemonika_from_parsing = random.choice(dict_fonems.loc[index, 'mnemonic_list'])
    else:
        mnemonika_from_parsing = 'Пока никто не придумал'

    # Поиск ближнего слова в словаре по Левенштейну
    mnemonika = difflib.get_close_matches(word_rus, dict['term'], n=1)

    if len(mnemonika) > 0:
        mnemonika = mnemonika[0]

    return [mnemonika, mnemonika_from_parsing]


# Функция поиска мема по произношению
parser = YandexImage()
def get_mem(mnemonika, word_ru):
    url_photo_mem = ''
    for item in parser.search(f'мем {mnemonika}'):
        url_photo_mem = item.url
        break
        # print(url_photo_mem)
    #
    # if not len(url_photo_mem):
    #     for item in parser.search(f'{mnemonika_from_parsing} мемчик'):
    #         url_photo_mem = item.url
    #         break
    #     print(url_photo_mem)
    #
    # if not len(url_photo_mem):
    #     for item in parser.search(f'{word_ru} мемчик'):
    #         url_photo_mem = item.url
    #         break
    #     print(url_photo_mem)

    return url_photo_mem


# Функция риспелинга английского слова в соотвествии с произношением
convertr = lytspel.Converter() # создаем обьект класса респелинг
def respell(word_eng):
    word_eng_respell = convertr.convert_para(word_eng)
    return word_eng_respell


# Функция поиска фото по русской транскрипции pixabay
# def get_photo_by_word(output_en):
#     res = requests.get(
#         'https://pixabay.com/api',
#         params={
#             'key': '21049512-19852da54db1fbfaebd9ccffc',
#             'q': f'<{output_en}>',
#             'lang': 'ru',
#             'per_page': 3,
#         }
#     )
#     image_urls = [img_data['largeImageURL'] for img_data in res.json()['hits']]
#     if len(image_urls) > 0:
#         image_urls = image_urls[0]
#     else:
#         image_urls = "Фото не нашлось"
#     return image_urls

# dict = pd.read_csv('/Users/antonzagorelskii/Desktop/Elbrus/Projects/Google API/rus_dict_50.csv', sep=';')

