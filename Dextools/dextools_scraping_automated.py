import argparse
import pandas as pd
import requests
import datetime
import os

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
    url = pair[1]

    options = Options()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=options)
    driver.get(url)
    page = 0 #page iterator
    all_results = [] #list for all rows

    #apply filtering for no bots
    filter_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//a[@class="d-inline-block wallet-filter-modal ng-tns-c169-3"]'))
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

    while page < pair[2]: #only temporary, change to True later

        page += 1

        if page%10 == 0:
            print('--- page:', page, '--- of ', pair[0])
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

    columns = ['datetime', 'buy_order', 'price_USD', 'price_native', 'amount_token', 'total_native', 'maker', 'etherscan_url']
    #dtypes={'datetime': 'datetime64[ns]'}#, 'amount_token': 'float'}
    df = pd.DataFrame(data, columns=columns)#, dtype = dtypes)

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S') #chnaging time to correct format
    df['buy_order'] = df['buy_order'].map({'buy': True, 'sell': False})
    df['price_USD'] = df['price_USD'].str.replace('$', '', regex=False) #removing $ sign from USD price string

    #list for subscript replacements
    replacements = [
    ('₂₀', 20*'0'),
    ('₁₉', 19*'0'),
    ('₁₈', 18*'0'),
    ('₁₇', 17*'0'),
    ('₁₆', 16*'0'),
    ('₁₅', 15*'0'),
    ('₁₄', 14*'0'),
    ('₁₃', 13*'0'),
    ('₁₂', 12*'0'),
    ('₁₁', 11*'0'),
    ('₁₀', 10*'0'),
    ('₉', 9*'0'),
    ('₈', 8*'0'),
    ('₇', 7*'0'),
    ('₆', 6*'0'),
    ('₅', 5*'0'),
    ('₄', 4*'0'),
    ('₃', 3*'0'),
    ('₂', 2*'0'),
    ('₁', 1*'0'),
    ('₀', 0*'0')
    ]

    #replacing subscripts with numbers 
    for subscripted, zeros in replacements:
        df['price_USD'] = df['price_USD'].str.replace(subscripted, zeros, regex=False)
        df['price_native'] = df['price_native'].str.replace(subscripted, zeros, regex=False)
        df['amount_token'] = df['amount_token'].str.replace(subscripted, zeros, regex=False)
        df['total_native'] = df['total_native'].str.replace(subscripted, zeros, regex=False)

    #changing dtype to floats
    df['price_USD'] = df['price_USD'].str.replace(",", "", regex=False).astype(float)
    df['price_native'] = df['price_native'].str.replace(",", "", regex=False).astype(float)
    df['amount_token'] = df['amount_token'].str.replace(",", "", regex=False).astype(float)
    df['total_native'] = df['total_native'].str.replace(",", "", regex=False).astype(float)


    return df


def save_dataframe_to_csv(dataframe, name): #saving dataframe to specific location
    # Create directory if it doesn't exist
    directory = os.path.join(os.getcwd(), 'scraped_data', name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Create filename with current time
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{name}_{time_str}.csv"
    filepath = os.path.join(directory, filename)
    
    # Save dataframe to CSV
    dataframe.to_csv(filepath, index=False)

def log_scraping_start():
    # Define the path to the logs file
    logs_path = os.path.join(os.getcwd(), "scraped_data", "logs.txt")
    dir_path = os.path.join(os.getcwd(), "scraped_data")

    # Create the scraped_data folder if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Open the logs file for appending
    with open(logs_path, "a") as f:
        # Write a separator line
        f.write("--------------\n")
        # Write the scraping start message with the current time
        f.write(f"Scraping starts at {current_time}\n")
    
    return current_time

def log_scraping_end(start_time):
    # Define the path to the logs file
    logs_path = os.path.join(os.getcwd(), "scraped_data", "logs.txt")

    # Get the end time and duration
    end_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    elapsed_time = datetime.datetime.strptime(end_time, '%Y-%m-%d_%H-%M-%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d_%H-%M-%S')

    # Open the logs file for appending
    with open(logs_path, "a") as f:
        # Write the scraping end message with the current time and duration
        f.write(f"Scraping ends at {end_time}, taking {elapsed_time}.\n")

def log_scraped_pair(df, name):
    # Define the path to the logs file
    logs_path = os.path.join(os.getcwd(), "scraped_data", "logs.txt")

    # Get the number of non-NA observations in the third column
    n = df.iloc[:, 2].count()

    # Get the first and last cells of the first column as datetime objects
    time1 = df.iloc[0, 0]
    time2 = df.iloc[-1, 0]

    # Open the logs file for appending
    with open(logs_path, "a") as f:
        # Write the message to the file
        f.write(f" - {name} scraped with {n} observations, ranging from {time1} to {time2}.\n")    

def log_scraping_fail(attempts, name, exception):        
    # Define the path to the logs file
    logs_path = os.path.join(os.getcwd(), "scraped_data", "logs.txt")

    # Open the logs file for appending
    with open(logs_path, "a") as f:
        # Write the scraping fail log and exception
        f.write(f"  --  Scraping {name} failed after {attempts} attempts.\n Exception: {exception}. \n")


if __name__ == '__main__':
    # Set up command line arguments with default values
    # parser = argparse.ArgumentParser()
    # parser.add_argument("arg1", type=int, nargs='?', default=1, help="the first argument (default 1)")
    # parser.add_argument("arg2", type=int, nargs='?', default=2, help="the second argument (default 2)")
    # args = parser.parse_args()
    # print(args.arg1, args.arg2)

    pair_list = [
                ("SHIB-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x811beed0119b4afce20d2583eb608c6f7af1954f", 50),
                ("HEX-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x55d5c232d921b9eaa6b37b5845e439acd04b4dba", 100),
                ("AGIX-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0xe45b4a84e0ad24b8617a489d743c52b84b7acebe", 40),
                ("OPTIMUS-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x8de7a9540e0edb617d78ca5a7c6cc18295fd8bb9", 70),
                ("SHIK-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x0b9f5cef1ee41f8cccaa8c3b4c922ab406c980cc", 60),
                ("INJ-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x1bdcebca3b93af70b58c41272aea2231754b23ca", 50),
                ("VOLT-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x487bfe79c55ac32785c66774b597699e092d0cd9", 200),
                ("MBOX-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x8fa59693458289914db0097f5f366d771b7a7c3f", 90),
                ("FLOKI-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x231d9e7181e8479a8b40930961e93e7ed798542c", 200),
                ("BabyDoge-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0xc736ca3d9b1e90af4230bd8f9626528b3d4e0ee0", 200)
                ]
    
    #loging the start of scraping
    start_time = log_scraping_start()

    #loop through the pairs to scrape
    for pair in pair_list:

        attempts = 10
        for i in range(attempts):

            try:
                #scraping data
                data = scrape_data(pair)

                #transforming data into pd.df
                df = data_transform_save(data, pair)

                #saving dataframe
                save_dataframe_to_csv(df, pair[0])

                #logging the saved data
                log_scraped_pair(df, pair[0])
                break
            
            #try to run the scrape 'attempts' times
            except Exception as e:
                if i < attempts - 1:
                    print(f"Attempt {i+1} failed with error: {e}")
                else:
                    log_scraping_fail(attempts, pair[0], e)
                    #not want to terminate program - commenting out the 'rise Exception' block
                    #raise Exception(f"Function failed after {attempts} attempts")
    
    #loging the end of scraping
    log_scraping_end(start_time)









#TODO list
# transform data to correct formats - to be continued, resolve the zeros in subscript
# transform into pd.df
# save dataframe with specific name   
# perhaps work out some tests?         