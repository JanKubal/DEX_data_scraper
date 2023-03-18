import argparse
import pandas as pd
import requests
import datetime

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

#define the scraping function
def scrape_data(pair):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    url = pair[1]

    options = Options()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=options)
    driver.get(url)

    page = 0 #page iterator
    all_results = [] #list for all rows

    #apply filtering for no bots
    filter_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//a[@class="d-inline-block wallet-filter-modal ng-tns-c168-3"]'))
    )
    #filter_button = driver.find_element(By.XPATH, '//a[@class="d-inline-block wallet-filter-modal ng-tns-c168-3"]')
    driver.execute_script("arguments[0].click();", filter_button) #opening filter window
    #time.sleep(0.5) #for some reason this is important?

    bot_checkbox = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//input[@id="bot-filtered"]'))
    )
    #bot_checkbox = driver.find_element(By.XPATH, '//input[@id="bot-filtered"]')
    driver.execute_script("arguments[0].click();", bot_checkbox) #ticking the checkbox
    #time.sleep(0.5)

    apply_filter_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//button[@class="btn btn-success ng-star-inserted"]'))
    )
    #apply_filter_button = driver.find_element(By.XPATH, '//button[@class="btn btn-success ng-star-inserted"]')
    driver.execute_script("arguments[0].click();", apply_filter_button) #applying filter
    time.sleep(0.5)

    while page < 3: #only temporary, change to True later

        page += 1

        if page%2 == 0:
            print('--- page:', page, '---')
        #print("before scrolling to view")
        # get table
        tableElement = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'ngx-datatable'))
        )
        # scroll into table view
        driver.execute_script("arguments[0].scrollIntoView();", tableElement)

        # scrolling through the table body to the bottom
        tableBodyelement = tableElement.find_element(By.TAG_NAME, "datatable-body-cell")
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight)", tableBodyelement)

        #print("after scrolling to view")
        
        rowWrapper = tableElement.find_elements(By.TAG_NAME, 'datatable-row-wrapper')
        #print("finding elements")
        all_results += [[cells[0].text,
                        cells[1].text,
                        cells[2].text,
                        cells[3].text,
                        cells[4].text,
                        cells[5].text,
                        cells[6].text,
                        cells[7].find_element(By.TAG_NAME, 'a').get_attribute('href')
        ] for row in rowWrapper for cells in [row.find_elements(By.TAG_NAME, 'datatable-body-cell')]]

        #moving to the next page                           
        #print("elements found")
        try:
            next_page = driver.find_element(By.XPATH, '//a[@aria-label="go to next page"]')

            #verify that the button is not disabled
            next_page_parent = next_page.find_element(By.XPATH, '..')
            if "disabled" in next_page_parent.get_attribute("class"):
                print("End of table")
                break

            #click on the button and move to next page
            #print("clicking next page")
            driver.execute_script("arguments[0].click();", next_page) 
            time.sleep(0.5)
            #print("next page clicked")
        except Exception as ex:
            print(f"Exception occured: {ex}")
            break
        
        #print("while loop end")

    driver.quit()  
    return all_results


def data_transform_save(data, pair):

    columns = ['datetime', 'type', 'price_USD', 'price_native', 'amount_token', 'amount_native', 'maker', 'etherscan_url']
    #dtypes={'datetime': 'datetime64[ns]'}#, 'amount_token': 'float'}
    df = pd.DataFrame(data, columns=columns)#, dtype = dtypes)

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')


    return df



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

        data = scrape_data(pair)

        print(len(data))
        for i in range(5):
            print(data[i])

        df = data_transform_save(data, pair)
        print(df.head())
        print(df.info())
        print(df.describe())









#TODO list
# transform data to correct formats - to be continued, resolve the zeros in subscript
# transform into pd.df
# save dataframe with specific name            