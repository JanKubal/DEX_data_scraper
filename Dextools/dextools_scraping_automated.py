import argparse
import pandas as pd
import requests
import datetime

#define the scraping function
def scrape_data(*args):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for name, url in args:
        # response = requests.get(url) #chatGPT code
        # if response.status_code == 200:
        #     data = response.content.decode('utf-8')
        #     df = pd.DataFrame({'Name': [name], 'Data': [data]})
        #     filename = f"{name}_{current_time}.csv"
        #     df.to_csv(filename, index=False)
        # else:
        #     print(f"Error: Failed to fetch data from {url}")
         print(name, url) #here continue with working scraping function



if __name__ == '__main__':
    # Set up command line arguments with default values
    # parser = argparse.ArgumentParser()
    # parser.add_argument("arg1", type=int, nargs='?', default=1, help="the first argument (default 1)")
    # parser.add_argument("arg2", type=int, nargs='?', default=2, help="the second argument (default 2)")
    # args = parser.parse_args()
    # print(args.arg1, args.arg2)

    pair_list = [("SHIB/WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x811beed0119b4afce20d2583eb608c6f7af1954f")]#, 
                #("HEX/WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x55d5c232d921b9eaa6b37b5845e439acd04b4dba")] #here continue and write all pairs


    #loop through the pairs to scrape
    for pair in pair_list:

        scrape_data(pair)