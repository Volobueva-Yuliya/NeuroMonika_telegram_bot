import logging
import common
from autotest_lib.client.common_lib.cros import chromedriver
with chromedriver.chromedriver() as chromedriver_instance:
    driver = chromedriver_instance.driver

import time
from selenium.webdriver import Chrome
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Настройка работы selenium
chrome_options = Options()
driver = webdriver.Chrome('chromedriver.exe')
# driver.set_window_size(1920, 1080)

# Вход на сайт и обход начального обучения 
url = "http://englspace.com/mnemo/"
driver.get(url)
# driver.get(url)

# tutorial_xpath = WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div[2]/div[1]/div[1]/div/div/div[3]/div/div/div/div[1]/i')))
# tutorial_xpath.click()
# time.sleep(5)

df = pd.DataFrame(columns=['english', 'transcription', 'russian', 'mnemonic_list'])


df_dictionary = pd.read_csv('kartaslovsent.csv', sep=';')

for i in range(df_dictionary.shape[0]):
    # Подать слово из словаря

    driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[2]/div/form/div[1]/input").send_keys(df_dictionary.loc[i, 'term'])
    time.sleep(2)
    find = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[2]/div/form/div[1]/div/input')
    find.click()

    # Проверить, что слово есть в мнемонике на данном сайте
    is_word_find = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[6]/div/div[1]/p[1]').text
    not_found = 'По вашему запросу ничего не найдено'
    if is_word_find != not_found:   
        table_xpath = '/html/body/div[1]/div[2]/div[2]/div[6]/div/div[2]/table/tbody/'
        time.sleep(5)
        rows = len(driver.find_elements_by_xpath(table_xpath + 'tr'))

        temp = []
        temp_translate = driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/div[2]/div[6]/div/div[2]/table/tbody/tr[1]/td/div[{2}]/div').text
        temp_translate = temp_translate.split()[0:2]
        temp_translate.append(df_dictionary.loc[i, 'term'])
        print(temp_translate)
        print('temp_translate size = ' + str(len(temp_translate)))
        # Получить список с мнемониками
        for row in range(1, rows):
            street = driver.find_element_by_xpath(
                table_xpath + "tr["+str(row)+"]").text
            temp.append(driver.find_element_by_xpath(f'/html/body/div[1]/div[2]/div[2]/div[6]/div/div[2]/table/tbody/tr[{row}]/td/div[3]/div').text)
        if not temp:
            temp.append(driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[6]/div/div[2]/table/tbody/tr/td/div[3]/div').text)
        temp_translate.append(temp)
        print('after finding')
        print(temp_translate)
        print('temp_translate size = ' + str(len(temp_translate)))
        df.loc[-1] = temp_translate
    time.sleep(5)
    driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[2]/div/form/div[1]/input").clear()
    

df.to_csv('test.csv', index=False)

# # Закрыть окно браузера
# # driver.close()
