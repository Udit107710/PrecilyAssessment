#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ChromeOptions
import pandas as pd
import time
import threading
from multiprocessing.pool import ThreadPool
import pickle


# In[2]:


class Scraper:
    def __init__(self):
        self.options =  ChromeOptions()
        self.prefs = {"profile.default_content_setting_values.notifications" : 2}
        self.options.add_experimental_option("prefs",self.prefs)
        self.options.add_argument("start-maximised")
        self.options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(options=self.options,executable_path='C:/Users/Udit/Downloads/chromedriver')
        self.wait = WebDriverWait(self.driver, 10)
        self.data = {'id': [], 'title': [], 'description': [], 'category': []}
        self.links = []
        self.temp = []
        
    def get_links_youtube(self,category):
        
        search_text = category
        if len(category.split(" ")) > 1:
            search_text = category.split(" ")[0]
            for text in category.split(" ")[1:]:
                search_text+= "+"+text
        print(search_text)
        self.driver.get('https://www.youtube.com/results?search_query='+search_text)
        try:
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@id='contents'and @class='style-scope ytd-item-section-renderer']")))
        except TimeoutException:
            self.driver.get('https://www.youtube.com/results?search_query='+category)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@id='contents'and @class='style-scope ytd-item-section-renderer']")))
            
            
        last_height = len(self.driver.find_element_by_xpath("//div[@class='style-scope ytd-item-section-renderer' and @id='contents']").find_elements_by_tag_name("ytd-video-renderer"))
        #print(last_height)
        while True:
            self.driver.find_element_by_tag_name('body').send_keys(Keys.END)
            time.sleep(3)
            self.driver.find_element_by_tag_name('body').send_keys(Keys.END)
            time.sleep(3)
            
            new_height = len(self.driver.find_element_by_xpath("//div[@class='style-scope ytd-item-section-renderer' and @id='contents']").find_elements_by_tag_name("ytd-video-renderer"))
            if new_height == last_height:
                print(new_height,last_height,"youtube stopped scrollign!")
                break
            last_height = new_height
            
        contents = self.driver.find_element_by_xpath("//div[@id='contents'and @class='style-scope ytd-item-section-renderer']").find_elements_by_xpath("ytd-video-renderer")    

        for content in contents:    
            url = content.find_element_by_id("video-title").get_attribute("href")
            #print(url)
            data_link = (url,category,'youtube')
            self.links.append(data_link)
            
            
    def get_links_dailymotion(self,category):
        self.driver.get('https://www.dailymotion.com/search/' + category + '/videos')
        try:
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='Grid Grid__grid___U2CEO']")))
        except TimeoutException:
            self.driver.get('https://www.dailymotion.com/search/' + category + '/videos')
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='Grid Grid__grid___U2CEO']")))
            
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to bottom
            self.driver.find_element_by_tag_name('body').send_keys(Keys.END)
            time.sleep(3)
            self.driver.find_element_by_tag_name('body').send_keys(Keys.END)
            time.sleep(3)
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print(new_height,last_height,"dailymotion stopped scrolling!")
                break
            last_height = new_height
            
            contents = self.driver.find_element_by_xpath("//div[@class='Grid Grid__grid___U2CEO']")
            values = contents.find_element_by_tag_name("div")
            section = values.find_elements_by_xpath("//section[@class='Video__wrap___2atEf Video__video___2Qq1K Video__smallHorizontal___2qSHD']")
            for video_data in section:
                url = video_data.find_element_by_tag_name("a").get_attribute("href")
                if url not in self.temp:
                    self.temp.append(url)
                    link_data = (url,category,'dailymotion')
                    self.links.append(link_data)
            
    
    def get_data(self):
        for data_link in self.links:
            link, category, source = data_link
            #print(data_link)
            if source == 'youtube':
                self.driver.get(link)
                time.sleep(2)
                try:
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string")))
                except :
                    self.driver.get(self.links[self.links.index(link) + 1])
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string")))
                time.sleep(3)
                self.data['id'].append(link.split("=")[-1])
                self.data['category'].append(category)
                self.data['title'].append(self.driver.find_element_by_xpath("//h1[@class='title style-scope ytd-video-primary-info-renderer']").text)
                self.data['description'].append(self.driver.find_element_by_xpath("//div[@class='style-scope ytd-video-secondary-info-renderer' and @id='description']").text)
                
            if source == 'dailymotion':
                self.driver.get(link)
                time.sleep(2)
                try:
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.Root__page___1mPV6")))
                except:
                    self.driver.get(self.links[self.links.index(link) + 1])
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div.Root__page___1mPV6")))
                time.sleep(5)
                content = self.driver.find_element_by_css_selector("div.Root__page___1mPV6")
                wait_1 = WebDriverWait(content,5)
                wait_1.until(EC.presence_of_element_located((By.CLASS_NAME,"VideoInfoDescription__descriptionText___RB9jX")))
                time.sleep(5)
                self.data['category'].append(category)
                self.data['id'].append(link.split("/")[-1])
                self.data['title'].append((content.find_element_by_class_name("VideoInfoTitle__videoTitle___11AcS")).text)
                self.data['description'].append((content.find_element_by_class_name("VideoInfoDescription__descriptionText___RB9jX")).text)


# In[3]:


youtube_scrapper = Scraper()
dailymotion_scrapper = Scraper()


# #### Multithreading to collect video link

# In[ ]:


categories = ['travel blog','science and technology', 'food', 'manufacturing', 'history','art and music']
for category in categories:
    t1 = threading.Thread(target=youtube_scrapper.get_links_youtube,args=(category,))
    t2 = threading.Thread(target=dailymotion_scrapper.get_links_dailymotion,args=(category,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


# with open('list_youtube.pkl','rb') as f:
#     youtube_scrapper.links = pickle.load(f)
# with open('list_daily.pkl','rb') as f:
#     dailymotion_scrapper.links = pickle.load(f)
# 
# youtube_scrapper.links[0:5]
# 
# dailymotion_scrapper.links[0:5]

# with open('list_youtube.pkl','wb') as f:
#     pickle.dump(youtube_scrapper.links,f)
# with open('list_daily.pkl','wb') as f:
#     pickle.dump(dailymotion_scrapper.links,f)

# #### Multithreading to collect data related to videos

# In[7]:


t3 = threading.Thread(target=youtube_scrapper.get_data,args=())
t4 = threading.Thread(target=dailymotion_scrapper.get_data,args=())
t3.start()
t4.start()
t3.join()
t4.join()


# In[ ]:





# In[8]:


data_1 = dailymotion_scrapper.data
data_2 = youtube_scrapper.data


# pprint(data_1)

# In[10]:


df_1 = pd.DataFrame.from_dict(data_1)
df_2 = pd.DataFrame.from_dict(data_2)


# In[11]:


df_1 = df_1.append(df_2,ignore_index=True)


# In[13]:


with open('data.pkl','wb') as f:
    pickle.dump(df_1,f)


# youtube_scrapper.data['id'] = []
# youtube_scrapper.data['title'] = []
# youtube_scrapper.data['category'] = []
# youtube_scrapper.data['description'] = []
