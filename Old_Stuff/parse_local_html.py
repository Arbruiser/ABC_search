# this program is simply parse the html codes
# with html file in the same folder

from bs4 import BeautifulSoup

with open("product_page.html", "r") as f:
    doc = BeautifulSoup(f, "html.parser")
print(doc.prettify()) # .prettify() function makes the outlines prettier
