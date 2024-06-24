# Read Json
import json
import pandas as pd


# Load Json file
json_file_path = './data/MELD/utterance-ordered-expanded.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data= json.load(file)

    dict={}

    # Select
    for key in data.keys():
        dict[key]= {}
        print("############Reading: ", key, "############")

        for dialogue_id, utterances in data[key].items():
            print("Dialogue ID: ", dialogue_id)
            dict[key][dialogue_id]= []
            for utterance_id, utterance in utterances.items():
                dict[key][dialogue_id].append(utterance["Speaker"]+":"+utterance["Utterance"] +"-->"+utterance["Sentiment"])
    
    # Save the expanded dialogues to a new JSON file
    with open('./data/MELD/dialogue.json', 'w', encoding='utf-8') as file:
        json.dump(dict, file, indent=4)

    
