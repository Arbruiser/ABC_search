#!/usr/bin/env python3

#the scripts scrapes a weather website and prints current temperature in Helsinki
import requests
from bs4 import BeautifulSoup

url = "https://www.bbc.com/weather/658225"
wpage = requests.get(url)
doc = BeautifulSoup(wpage.text, "html.parser")
#print("len of the html is:", len(doc))

#temperature = doc.find_all(string="-9°")
#print(temperature)
#parent = temperature[0].parent
#span = parent.find("wr-value--temperature--c")
#print(span)

temperature_element = doc.find('span', {"class": "wr-value--temperature--c"})
print("Current temperature in Helsinki is:", temperature_element.text)

#<span class="wr-value--temperature--c">-9°</span>
