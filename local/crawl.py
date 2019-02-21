#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding: utf8 
# coding: cp949
"""
Created on Tue Jul 24 15:21:03 2018

@author: Bohyun
"""
##메소드 바로 사용하기 위해서 from~ import * 사용

import requests as rq
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, parse_qs
import pandas as pd


import csv
#from models.commends import Commends
#from database import db_session

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import re
import time

# HTTP GET Request
#req = rq.get('https://play.google.com/store/apps/details?id=com.ludia.jurassicworld')
# HTML 소스 가져오기
#html = req.text
# BeautifulSoup 으로 html 소스를 python 객체로 변환하기
#soup = BeautifulSoup(html, 'html.parser')

##파이썬은 인터프리터 명령어로 패싱되어 실행되어서  자동으로 실행되는 메인함수가 없다
##__name__: 현재 모듈의 이름을 담고있는 내장 변수 -> testweb.py같이 이 모듈이 직접 실행되는 경우에만 __name__ 이 __main__으로 실행 

driver = webdriver.Chrome(r'C:\Users\Bohyun\.spyder-py3\project\chromedriver.exe')
driver.get("https://play.google.com/store/apps/details?id=com.cmcm.arrowio&showAllReviews=true")
driver.implicitly_wait(3)

#encoding='CP949'
"""
f=open("texttest.txt", 'w', encoding='UTF-8')
file = open('texttest.txt','a',encoding='utf-8')

wr=csv.writer(file, delimiter='\t')
"""

    
#엑셀 파일 만들기
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

        print("No.",count)
  #     f.write(str(count+1))
   #     f.write(";")    
        ##게임 이름
        app_name = soup.select_one('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > c-wiz > c-wiz > div > div.D0ZKYe > div > div.sIskre > c-wiz > h1 > span')
        print("[게임이름]: ", app_name.string)
   #     f.write(app_name.string)
   #     f.write("\t")  
        ##게임 카테고리
        app_categ = soup.select_one('#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div.JNury.Ekdcne > div > c-wiz > c-wiz > div > div.D0ZKYe > div > div.sIskre > div.jdjqLd > div.ZVWMWc > div > span:nth-of-type(2)')
        genre = app_categ.find("a", itemprop="genre")
        print("[카테고리]: ", genre.string)   
    #    f.write(genre.string)
    #    f.write("\t")
        ##유용함
        useful = com.select_one('div.xKpxId.zc7KVe > div.YCMBp.GVFJbb > div > span > div > content > span > div')
        print("[유용함]: ", useful.string)
    #    f.write(useful.string)
    #    f.write("\t")
        ## 댓글 내용
        
        content = com.select_one('div.UD7Dzf > span:nth-of-type(1)')
        print("[내용]: ",content.string)

        ##댓글 작성자
        writer = com.select_one('div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > span')
        print("[작성자]: ", writer.string)
        
#        f.write(writer.string)
#        f.write("\t")
#        f.write(str(content.string))
#        f.write("\t")
        ##반복문을 통하여 가져오기
        #wr.writerow([content.string])

        ## 댓글 평점
        rating = com.select_one('div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > div > span.nt2C1d > div > div')
        rat = rating.find_all("div", class_='vQHuPe bUWb7c')
        length=len(rat)
        print("[평점]: ", length)   
     #   f.write(str(length))
     #   f.write("\t")
        ##댓글 날짜
        date = com.select_one("div.xKpxId.zc7KVe > div.bAhLNe.kx8XBd > div > span.p2TkOb")
        print("[날짜]: ", date.string)
     #   f.write(date.string)
     #   f.write("\t")
        count+=1
    #    f.write("\n")
        ##댓글 반응
        '''
        if length > 2.5:
            response = 1 #긍정
            print("[반응]: ", response)
        else:
            response = 0 #부정
            print("[반응]: ", response)
        '''
 #       f.write(str(response))
 #       f.write("\n")
        temp=[]
        with open("애로우.txt", 'a+', encoding='UTF-8') as out:  
            #작성자!!
            out.write(str(writer.string))
            out.write("\t")          
            #list(str(content.string))
            #line.rstrip('\n')
            #내용!!
            temp=(str(content.string))          
            ##split 으로 개행문자 나누고 한 줄씩 쓰기
        
            temp_array=temp.split('\n')
            temp_len=len(temp_array)
            for i in range(temp_len):
                temp_array[i].rstrip('\n')
                out.write(temp_array[i])
                
                print('temp_array',i,': ',temp_array[i])
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

    
  