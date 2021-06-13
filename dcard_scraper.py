# -*- coding: utf-8 -*-
import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time

dcard_url_1 = "https://www.dcard.tw/f/mood"     #心情
#dcard_url_2 = "https://www.dcard.tw/f/relationship"    #感情
#dcard_url_3 = "https://www.dcard.tw/f/talk"    #閒聊
#dcard_url_4 = "https://www.dcard.tw/topics/%E6%B8%AF%E6%BE%B3%E7%89%88"    #港澳版
dcard_urls = [dcard_url_1] #, dcard_url_2, dcard_url_3, dcard_url_4]

options = Options()
options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
driver_url=r"C:\Users\Wesley Hung\Desktop\geckodriver-v0.29.1-win32\geckodriver.exe"
browser = webdriver.Firefox(options=options, executable_path=driver_url)

for dcard_url in dcard_urls:
    browser.get(dcard_url)
    time.sleep(5)
    browser.find_element(By.LINK_TEXT, "恐怖網友 交友軟體的可怕").click()
    i = 0


    df = pd.DataFrame(columns = ['title', 'post_content', 'post_comment', 'likes'])

    try:
        for i in range(500):
            browser.find_element(By.CSS_SELECTOR, ".llPrcG").click()
            soup = BeautifulSoup(browser.page_source, features="lxml")
            # get the post title
            title = ""
            for ele in soup.select('.sc-1932jlp-0.cqaWIE'):
                title = ele.text
                print("title: ", title)

            # get the post content
            content = ""
            for ele in soup.select(".sc-1npvbtq-0.gfjrnD .phqjxq-0.fQNVmg"):
                content = ele.text
                #print("content: ", content)
                break

            time.sleep(5)
            # get the post comment
            comments=[]
            likes=[]
            comment_text = soup.select("#comment-anchor .pj3ky0-0.cpOUHp .phqjxq-0.fQNVmg")
            for c in comment_text:
                if (c.text != "已經刪除的內容就像 Dcard 一樣，錯過是無法再相見的！"):
                    comments.append(c.text)
                    #print("comment: ", c.text)

            comments_like = soup.select(".jt7qse-1.lhEwzj")
            for cl in comments_like:
                likes.append(cl.text)
                #print("# of like: ", cl.text)


            small_df = pd.DataFrame(index=np.arange(len(comments)), columns=['title', 'post_content', 'post_comment', 'likes'])
            small_df['title'] = title
            small_df['post_content'] = content
            small_df['post_comment'] = comments
            small_df['likes'] = likes
            df = df.append(small_df, ignore_index=True)

        print(df)

    finally:
        browser.close()

with pd.ExcelWriter('dcard_df.xlsx', mode='a') as writer:
    df.to_excel(writer, index=False)
