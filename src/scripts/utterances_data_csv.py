# Python script to generate a CSV file from the Utterances data
import pandas as pd
import json
import glob

def main():
    # Read Json File
    with open('data/MELD/utterance-ordered.json') as f:
        full_data = json.load(f)
    print("Data loaded successfully")


    for data_category in ['train','val','test']:
        print(f'Processing {data_category} data')

        # List all JSON files to merge
        json_files = glob.glob(f'data/MELD/raw-texts/{data_category}/*.json')

        # List to hold data
        data_list = [] 

        # Loop through each json
        for file in json_files:
            print(f'Processing {file}')
            with open(file, 'r',encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data_list)

        # Save the DataFrame to a CSV file
        save_path=f'data/MELD/{data_category}_utterances.csv'
        df.to_csv(save_path, index=False)

        print(f'{save_path} file saved successfully')



    pass
if __name__ =="__main__":

    print("This Script is used to generate a CSV file from the Utterances data in Json files.")
    main()


# python .\src\scripts\utterances_data_csv.py