# this program is simply parse the html codes
# with html file in the same folder

from bs4 import BeautifulSoup

with open("product_page.html", "r") as f:
    doc = BeautifulSoup(f, "html.parser")

tag = doc.title # find info based on tags, the example here is <title></title>
tag.string = "Welcome page" # this step is to modify the original contain of tag

print(tag.string) # .string is only extracting the characters without the tag
print(tag) # you could verify it by tag whether the title has been modified
print(doc.prettify()) # or print the whole document

tags = doc.find_all("p") # find all the p tags and shows what inside the p tags
print(tags)

tags_1 = doc.find_all("p")[0] # access to the first p tags with an index[0]
print(tags_1)
print(tags_1.find_all("br")) # find br tag within the first p tag