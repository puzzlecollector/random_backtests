from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import requests
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import time


#browser = webdriver.Firefox()#Chrome('./chromedriver.exe')
HOME_PAGE_URL = "https://dstreet.io/blockchain/news/money-market/"
LOAD_MORE_BUTTON_XPATH = '//*[@id="tdi_53_212"]/div/div[1]/div/div[3]/div[3]'

driver = webdriver.Chrome('./chromedriver')
driver.get(HOME_PAGE_URL)

while True:
    try:
        print("locating load more button...")
        loadMoreButton = driver.find_element_by_css_selector('div.td-load-more-wrap')
        time.sleep(2)
        loadMoreButton.click()
        time.sleep(2)
    except Exception as e:
        print(e)
        break
print("Complete")
time.sleep(10)
driver.quit()
