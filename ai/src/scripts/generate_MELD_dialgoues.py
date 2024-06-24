import json
import pandas as pd
import re
import sys



# Load Json file
json_file_path = './data/MELD/utterance-ordered.json'

generated_dialogues = {}

with open(json_file_path, 'r', encoding='utf-8') as file:
    data_type = json.load(file)


    for key in data_type.keys():
        print("############Reading: ", key, "############")
        # Create a new dictionary for the generated dialogues Type
        generated_dialogues[key] = {}

        # Load the CSV file into a dictionary
        csv_file_path = './data/MELD/' + key + '_utterances.csv'

        # Step 1: Load the CSV file into a dictionary
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            # Read the CSV file using pandas
            df = pd.read_csv(file)

        for dialogue_id, utterances in data_type[key].items():
            # Create a new list for the generated utterances

            generated_dialogues[key][dialogue_id]= {}


            for utterance_id in utterances:

                # Using regular expression to find all numbers
                numbers = re.findall(r'\d+', utterance_id)

                if len(numbers) >= 2:
                    dialogue_idx = int(numbers[0])
                    utterance_idx = int(numbers[1])
                    print("Reading Dialogue: ", dialogue_idx, "Utterance: ", utterance_idx)
                else:
                    print("Could not find two numbers in the text.")

                selected_row=(df[(df['Dialogue_ID'] == dialogue_idx) & (df['Utterance_ID'] == utterance_idx)])

                json_data = selected_row.to_json(orient='records')
                json_data=json.loads(json_data)[0]

                generated_dialogues[key][dialogue_id][utterance_id] = json_data

                # Get the text for the utterance_id
                # utterance_text = df[df['Dialogue_ID'] == dialogue_idx][df['Utterance_ID'] == utterance_idx ]

                # # Convert to JSON
                # generated_dialogues[key][dialogue_id][utterance_id] = utterance_text


                # utterance_text = utterance_text.to_json(orient='records')
                # print(utterance_text)

# Save the expanded dialogues to a new JSON file
expanded_json_file_path = './data/MELD/utterance-ordered-expanded.json'
with open(expanded_json_file_path, 'w', encoding='utf-8') as file:
    json.dump(generated_dialogues, file, ensure_ascii=False, indent=4)
