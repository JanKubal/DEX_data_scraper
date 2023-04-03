import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def load_csv_files(folder_path):
    df_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            name = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path).drop_duplicates().reset_index()
            df_dict[name] = df
    return df_dict

def concatenate_dataframes(df_dict):
    # Concatenate DataFrames
    df_list = list(df_dict.values())
    df_concatenated = pd.concat(df_list, axis=0)

    # Ensure datetime column is sorted
    df_concatenated.sort_values(by=["datetime", "index"], ascending=[False, True], inplace=True)
    df_concatenated.drop(columns=['index'], inplace=True)
    df_concatenated.drop_duplicates(inplace=True)

    return df_concatenated

def save_dataframe_to_csv(dataframe, name): #saving dataframe to specific location
    # Create directory if it doesn't exist
    directory = os.path.join(os.getcwd(), 'complete_data', name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Create filename with current time
    #time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #filename = f"{name}_complete_{time_str}.csv" #time not needed anymore
    filename = f"{name}_complete.csv"
    filepath = os.path.join(directory, filename)
    
    # Save dataframe to CSV
    dataframe.to_csv(filepath, index=False)



if __name__ == "__main__":
    pair_list = [
            ("SHIB-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x811beed0119b4afce20d2583eb608c6f7af1954f", 50),
            ("HEX-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x55d5c232d921b9eaa6b37b5845e439acd04b4dba", 100),
            ("AGIX-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0xe45b4a84e0ad24b8617a489d743c52b84b7acebe", 40),
            ("OPTIMUS-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x8de7a9540e0edb617d78ca5a7c6cc18295fd8bb9", 70),
            ("SHIK-WETH", "https://www.dextools.io/app/en/ether/pair-explorer/0x0b9f5cef1ee41f8cccaa8c3b4c922ab406c980cc", 60),
            ("INJ-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x1bdcebca3b93af70b58c41272aea2231754b23ca", 60),
            ("VOLT-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x487bfe79c55ac32785c66774b597699e092d0cd9", 200),
            ("MBOX-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x8fa59693458289914db0097f5f366d771b7a7c3f", 90),
            ("FLOKI-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0x231d9e7181e8479a8b40930961e93e7ed798542c", 180),
            ("BabyDoge-WBNB", "https://www.dextools.io/app/en/bnb/pair-explorer/0xc736ca3d9b1e90af4230bd8f9626528b3d4e0ee0", 180)
            ]
    

    for pair in pair_list:
        folder_path = f"D:/Dokumenty/Vejška/Magisterské studium/DIPLOMKA/Code_and_Data/Data_scraping/DEX_data_scraper/scraped_data/{pair[0]}"

        print(pair[0])

        df_dict = load_csv_files(folder_path)
        df_concatenated = concatenate_dataframes(df_dict)

        print(len(df_concatenated))

        save_dataframe_to_csv(df_concatenated, pair[0])
