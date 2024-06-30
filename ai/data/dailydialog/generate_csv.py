import pandas as pd

def main():

    dialogues_file_path = "./data/dailydialog/train/dialogues_train.txt"
    # Open the file with UTF-8 encoding explicitly
    with open(dialogues_file_path, 'r', encoding='utf-8') as file:
        data_txt = file.readlines()

    # Open file for act
    acts_file_path = "./data/dailydialog/train/dialogues_act_train.txt"
    with open(acts_file_path, 'r', encoding='utf-8') as file:
        data_acts = file.readlines()

        assert len(data_txt) == len(data_acts) # Check if the number of dialogues and acts are the same

    # Open file for emotion
    emotions_file_path = "./data/dailydialog/train/dialogues_emotion_train.txt"
    with open(emotions_file_path, 'r', encoding='utf-8') as file:
        data_emotions = file.readlines()
        assert len(data_txt) == len(data_emotions)

    print("Data loaded successfully!")
    print("Total Number of dialogues: ", len(data_txt))


    # Create a pd data frame
    df = pd.DataFrame({'text': data_txt, 'act': data_acts, 'emotion': data_emotions})

    # Save the data frame to a csv file
    df.to_csv("./data/dailydialog/train/train.csv", index=False)
    print("Data saved to csv successfully! at", "./data/dailydialog/train/train.csv")



if __name__ =="__main__":
    main()