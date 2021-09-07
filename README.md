# NeuroMonika_telegram_bot

NeuroMonika - то бот, который знает как быстро и эффективно выучить английские слова.  
Ссылка https://t.me/NeuroMonika_bot

## Идея
В школе английский язык изучается на протяжении многих лет, но не каждый может похвастаться отличным владением языка во взрослой жизни.
А все почему? Как правило, слова с неизвестным, абстрактным значением запомнить большинству людей сложно. Если такое слово «зазубрить», то оно исчезает из памяти через несколько дней.  
<img align="right" src="https://github.com/Volobueva-Yuliya/NeuroMonika_telegram_bot/blob/main/jmg/mnemonics.png" width="400" />  
Например, перевод слова ПАЛЬТО. В английском языке это слово coat. Можно повторить раз 10 и запомнить хотя бы на ближайшие 10 минут. Скучно, неэффективно...  
Поэтому для прочного и одновременно лёгкого запоминания следует наполнить слово содержанием — чем-то, что связано с конкретным ярким визуальным образом. В основе этой техники лежит мнемоника  
Вернемся к ПАЛЬТО. Пальто coat созвучно со словом КОТ. Мы получили с вами ассоциативное словосочетание. И чтобы закрепить результат - обратимся к визуальному образу. Классно, не правда ли. И действительно, как показывает практика, мнемотехника помогает надолго запоминать много новых слов.  
  
  
## Этап разработки
<img align="right" src="https://github.com/Volobueva-Yuliya/NeuroMonika_telegram_bot/blob/main/jmg/realization.png" width="400" />  

1. Перевод слова (Google Translate API)  
2. Произношение - руинглиш (библиотеки: lytspel, pronounciation)  
3. Парсинг сайта Englspace.ru (Selenium)  
4. Поиск или генерация мнемоники?  
Это стало основным вопросом в процессе работы над проектом. Паралельно велась разработка в двух направления: использование и анализ алгоритма "расстояние Левенштейна", а также обучение рекуррентной нейроннной сети "Spell checker". После того, как были получены первые результаты, стало очевидно, что алгоритмический способ показал себя лучше. На нем и остановились. 

5. Визуальный образ. Парсинг yandex image mem.   
  
## Команда
<img align="right" src="https://github.com/Volobueva-Yuliya/NeuroMonika_telegram_bot/blob/main/jmg/team.png" width="400" />  

Мой вклад в проект:
- идея проекта  
- обучение модели проверки правописания  
- парсинг yandex image mem.
- установка бота на сервер  




В настоящий момент бот запущен и находится в статусе тестирования. Если хотите внести вклад в развитие проекта, пишите мне.  
Спасибо за внимание!
