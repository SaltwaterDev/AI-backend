# -*- coding: utf-8 -*-
from selenium import selenium
from selenium import webdriver
from selenium.webdriver.common.exception import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup

lihkg_url = "https://lihkg.com/category/30"
dcard_url = "https://www.dcard.tw/f/relationship"

browser = webdriver.Firefox()
browser.get(lihkg_url)
browser.find_element(By.CSS_SELECTOR, ".wQ4Ran7ySbKd8PdMeHZZR:nth-child(1) .\\_2A_7bGY9QAXcGu1neEYDJB").click()
browser.close() 