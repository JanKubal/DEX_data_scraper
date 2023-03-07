import requests
from bs4 import BeautifulSoup

# specify the URL of the page you want to scrape
url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'

# send a GET request to the URL
response = requests.get(url)

# parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# find the main content of the page by looking for the first <div> tag with class 'mw-parser-output'
div_content_text = soup.find("div", {"class":"mw-body-content mw-content-ltr"})
content_div = div_content_text.find('div', {'class': 'mw-parser-output'})

# get all the text content from the page by concatenating the text of all <p> tags within the content div
text_content = ''
for p in content_div.find_all('p'):
    text_content += p.get_text()

# print the text content
print(text_content)