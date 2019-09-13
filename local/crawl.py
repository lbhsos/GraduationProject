#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding: utf8 
# coding: cp949
"""
Created on Tue Jul 24 15:21:03 2018

@author: Bohyun
"""
import requests as rq
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, parse_qs
import pandas as pd


import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import re
import time

driver = webdriver.Chrome(r'C:\Users\Bohyun\.spyder-py3\project\chromedriver.exe')
driver.get("https://play.google.com/store/apps/details?id=com.cmcm.arrowio&showAllReviews=true")
driver.implicitly_wait(3)

def scroll():
    while True:
        global last_height
        global soup
        global flag
        SCROLL_PAUSE_TIME = 1
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            find_button()
            if (flag==1):
                soup = BeautifulSoup(driver.page_source, 'lxml')
                break
        
        last_height = new_height
                    
def find_button():
        ##더보기 버튼  
        global flag
        global soup
        try:
           driver.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div[1]/div/div/div/div[2]/div[2]/div').send_keys(Keys.ENTER) 
        except:
            flag=1
            pass
        else:
            flag=0
            scroll()

            


def crawl():
    global file
    global wr
    global pFlag
    global count
    global flag
    global page
    global soup
    global last_height
    global section ##더보기 버튼
    pFlag=1
    i=0
     ##댓글 전체 block
    res = soup.select('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > div > div > div > div > div > div > div.d15Mdf.bAhLNe') 
    for com in res: 
        ##게임 이름
        app_name = soup.select_one('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > c-wiz > c-wiz > div > div.D0ZKYe > div > div.sIskre > c-wiz > h1 > span')
        ##게임 카테고리
        app_categ = soup.select_one('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > c-wiz > c-wiz > div > div.D0ZKYe > div > div.sIskre > div.jdjqLd > div.ZVWMWc > div > span:nth-of-type(2)')
        genre = app_categ.find("a", itemprop="genre")
        ##유용함
        useful = com.select_one('div.xKpxId.zc7KVe > div.YCMBp.GVFJbb > div > span > div > content > span > div')
        ## 댓글 내용   
        content = com.select_one('div.UD7Dzf > span:nth-of-type(1)')
        ##댓글 작성자
        writer = com.select_one('div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > span')
        ## 댓글 평점
        rating = com.select_one('div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > div > span.nt2C1d > div > div')
        rat = rating.find_all("div", class_='vQHuPe bUWb7c')
        length=len(rat)
        ##댓글 날짜
        date = com.select_one("div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > div > span.p2TkOb")
        count+=1
        temp=[]
        with open("애로우.txt", 'a+', encoding='UTF-8') as out:  
            #작성자!!
            out.write(str(writer.string))
            out.write("\t")          
            temp=(str(content.string))          
            temp_array=temp.split('\n')
            temp_len=len(temp_array)
            for i in range(temp_len):
                temp_array[i].rstrip('\n')
                out.write(temp_array[i])
                i+=1
            i=0
            #out.write(temp)
            out.write("\t")
            #평점
            out.write(str(length))  
            out.write("\t")
            #유용함
            out.write(useful.string)  
            out.write("\t")
            #날짜
            out.write(date.string)  
            out.write("\t")
            #카테고리
            out.write(genre.string)  
            out.write("\t")
            out.write("\n")
            

            

if __name__ == "__main__":
    count=0
    flag=0
    section=0
    last_height=0
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    pFlag=0
    print('commend collecting crawler')
    app_name = soup.select_one('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > c-wiz > c-wiz > div > div.D0ZKYe > div > div.sIskre > c-wiz > h1 > span')
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll()
    crawl()
    print("댓글 전체 수: ", count)

    
  