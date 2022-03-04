# ppomppu 웹스크래핑 

from selenium import webdriver 
import requests
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import time

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("window-size=1920,1280")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    "AppleWebKit/537.36 (KHTML, like Gecko)"
    "Chrome/78.0.3904.108 Safari/537.36"
)

driver = webdriver.Chrome("./chromedriver", options=options)


texts = [] 

for i in range(2000, 3000):
    print("Collecting from page {}...".format(i))
    url = "https://www.ppomppu.co.kr/zboard/zboard.php?id=bitcoin&page=" + str(i) + "&divpage=1"
    driver.get(url) 
    list0 = driver.find_elements_by_css_selector("tr.list0 > td:nth-child(3) > a:nth-child(2)") 
    list1 = driver.find_elements_by_css_selector("tr.list1 > td:nth-child(3) > a:nth-child(2)")  
    arr = list0 + list1 

    links = [] 

    for x in arr: 
        link = x.get_attribute('href') 
        links.append(link)
    
    for link in tqdm(links):
        try:
            driver.get(link)
            time.sleep(0.1)
            elem = driver.find_element_by_css_selector("td.board-contents")
            texts.append(elem.text)
        except: 
            continue 
    

    time.sleep(1)
    
ppompu_df = pd.DataFrame(texts) 
ppompu_df = ppompu_df.rename(columns={0:'Text'}) 
ppompu_df['Source'] = '뽐뿌'
ppompu_df.to_csv("ppomppu3.csv",index=False)


