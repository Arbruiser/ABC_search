# this program is simply parse the html codes
# with html file in the same directory

from bs4 import BeautifulSoup
import requests

url = "https://www.amazon.com/dp/B08DWD38VX?ref_=Oct_DLandingS_D_87e65655_NA"

result = requests.get(url)
print(result.text) # exam the file
doc = BeautifulSoup(result.text, "html.parser")
print(doc.prettify()) # you can see now the page has been grabbed by you

prices = doc.find_all(text="$")
print(prices)
parent = prices[0].parent
print(parent) # exam the file to see if the price is appear
product_price = parent.find("span", {"class": "a-price-symbol"})
print("Current price of this product is:", product_price)

