import requests

import pandas as pd
import json

# specify the JSON file
url = 'https://www.dextools.io/chain-ethereum/api/Uniswap/history/candles?sym=usd&span=month&pair=0x4e68ccd3e89f51c3074ca5072bbac773960dfa36&ts=1677020400000&v=1662994304076&res=15m&timezone=1'

url = 'https://www.dextools.io/chain-ethereum/api/Uniswap/history/candles?sym=usd&span=month&pair=0x4e68ccd3e89f51c3074ca5072bbac773960dfa36&ts=1627772400000&v=1662994304076&res=15m&timezone=1'


#headers for the json, kudos ChatGPT
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Referer": "https://www.dextools.io/app/en/ether/pair-explorer/0x4e68ccd3e89f51c3074ca5072bbac773960dfa36"
}

# send a GET request to the URL
response = requests.get(url, headers=headers)

#print(response.status_code)
#print(response.json())

#print(response.json()["data"])

#data_dict = json.loads(response.json())
df = pd.DataFrame.from_dict(response.json()["data"])#, orient='index')

print(df.candles[1])
print(df.shape)