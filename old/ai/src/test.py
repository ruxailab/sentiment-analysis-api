import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_checkpoint='roberta-base'

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_checkpoint, num_labels=5
# )

# print(model)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


def prepare_input(dialogue, model_checkpoint, speaker_mode="upper", num_past_utterances=0, num_future_utterances=0):
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # print(tokenizer)
    # sys.exit()
    # max_model_input_size = tokenizer.max_model_input_sizes[model_checkpoint]
    max_model_input_size = tokenizer.model_max_length
    num_truncated = 0

    print("max_model_input_size",max_model_input_size) # 512
    # sys.exit()

    inputs = []
    ues = [
        {"Utterance": utt, "Emotion": None}  # Assuming `utt` is a string in the dialogue list
        for utt in dialogue
    ]

  
    num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues]
    print(num_tokens) #[8,10,6,10]
    # sys.exit()

    for idx, ue in enumerate(ues):
        indexes = [idx]
        indexes_past = [
            i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
        ]
        indexes_future = [
            i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
        ]

        offset = 0
        if len(indexes_past) < len(indexes_future):
            for _ in range(len(indexes_future) - len(indexes_past)):
                indexes_past.append(None)
        elif len(indexes_past) > len(indexes_future):
            for _ in range(len(indexes_past) - len(indexes_future)):
                indexes_future.append(None)

        for i, j in zip(indexes_past, indexes_future):
            if i is not None and i >= 0:
                indexes.insert(0, i)
                offset += 1
                if (
                    sum([num_tokens[idx_] for idx_ in indexes])
                    > max_model_input_size
                ):
                    del indexes[0]
                    offset -= 1
                    num_truncated += 1
                    break
            if j is not None and j < len(ues):
                indexes.append(j)
                if (
                    sum([num_tokens[idx_] for idx_ in indexes])
                    > max_model_input_size
                ):
                    del indexes[-1]
                    num_truncated += 1
                    break

        utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

        if num_past_utterances == 0 and num_future_utterances == 0:
            assert len(utterances) == 1
            final_utterance = utterances[0]

        elif num_past_utterances > 0 and num_future_utterances == 0:
            if len(utterances) == 1:
                final_utterance = "</s></s>" + utterances[-1]
            else:
                final_utterance = (
                    " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                )

        elif num_past_utterances == 0 and num_future_utterances > 0:
            if len(utterances) == 1:
                final_utterance = utterances[0] + "</s></s>"
            else:
                final_utterance = (
                    utterances[0] + "</s></s>" + " ".join(utterances[1:])
                )

        elif num_past_utterances > 0 and num_future_utterances > 0:
            if len(utterances) == 1:
                final_utterance = "</s></s>" + utterances[0] + "</s></s>"
            else:
                final_utterance = (
                    " ".join(utterances[:offset])
                    + "</s></s>"
                    + utterances[offset]
                    + "</s></s>"
                    + " ".join(utterances[offset + 1 :])
                )
        else:
            raise ValueError
        print(final_utterance)
        
        input_ids_attention_mask = tokenizer(final_utterance, padding='max_length', max_length=max_model_input_size, truncation=True)
        input_ids = input_ids_attention_mask["input_ids"]
        attention_mask = input_ids_attention_mask["attention_mask"]

        input_ = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        inputs.append(input_)

    return inputs


import torch

def predict(dialogue, model, tokenizer, speaker_mode="upper", num_past_utterances=0, num_future_utterances=0):
    model.eval()
    inputs = prepare_input(dialogue, model_checkpoint, speaker_mode, num_past_utterances, num_future_utterances)

    # print("inputs",inputs)
    # sys.exit()
    
    
    all_input_ids = torch.tensor([item['input_ids'] for item in inputs])
    all_attention_mask = torch.tensor([item['attention_mask'] for item in inputs])
    # print(all_input_ids)
    # sys.exit()

    with torch.no_grad():
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
        # print(outputs)
        # sys.exit()
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()
    
    return predictions



def map_predictions_to_emotions(predictions, id2emotion):
    return [id2emotion[pred] for pred in predictions]

# Load the id2emotion mapping
# id2emotion = {0: "neutral", 1: "joy", 2: "surprise", 3: "anger", 4: "sadness", 5: "disgust", 6: "fear"}
id2emotion = {0:"negative", 1:"neutral", 2:"positive"}
# id2emotion = {0:"negative", 1:"positive"}
# emotion2id = {
#         "neutral": 0,
#         "joy": 1,
#         "happiness": 1,
#         "excited": 1,
#         "surprise": 2,
#         "anger": 3,
#         "frustration": 3,
#         "sadness": 4,
#         "disgust": 5,
#         "fear": 6,
#     }

# id2emotion = {idx: emotion for idx, emotion in enumerate(emotions)}

# Example dialogue input
# dialogue = [
#     "Hello, how are you?",
#     "I'm doing well, thank you!",
#     "What about you?",
#     "I'm great, thanks for asking!",
#     "Have you tied Sportify before?",
#     "Yes, I love Spotify. It's my favorite music app!",
#     "How can you rate Sportify?",
#     "I don't like it.",
# ]

# dialogue = [
#     "I don't know what to say anymore. Hi Basma. Let's just finish maybe yes.",
#     "Or you will talk about Spotify.",
#     "I really like Spotify. I can live without it, right? You agree? You were telling me to say yes. I am talking to someone that is beside me. So Spotify is amazing. They have really good music there and really good recommendations. I really like it. You don't like it? No. Okay, I like it. Only my opinion matters. Okay. It's my word, my rules. But I think should be completely free. That's it.",
#     "Okay, so let's finish the test.",
#     "Yes.",
#     "And that's it.",
#     "I need to save and exit."
# ]

dialogue = [
    "Did you liked the movie?",
    "Yes, I loved it.",
    "Did you hate the movie?",
    "No,I did not like it.",
    # "No, I enjoyed it" # Positive
    # "I don't hate it" # Positive
    "Did you enjoy the movie?",
    "It is confusing.",
    "I hated that movie.",
]

# dialogue=['Did you like the party ? ',
#  ' Not particularly . I was rather disappointed . ',
#  ' Why ? What had you expected ? ',
#  ' I expected it to be much more exciting . ']

# dialogue = [
#     "Oh my God, oh my God! Poor Monica!",
#     "What, what, what?!",
#     "What?!",
#     "He was with her when he wrote this poem.",
#     "Look,  'My vessel so empty with nothing inside.",
#     "Now that I've touched you, you seem emptier still.'",
#     "He thinks Monica is empty, she is the empty vase!",
#     "Oh, totally. Oh, God, oh, she seemed so happy too.",
#     "Done."]

            # "Phoebe:Oh my God, oh my God! Poor Monica!-->negative",
            # "Chandler:What, what, what?!-->negative",
            # "Phoebe:What?!-->negative",
            # "Phoebe:He was with her when he wrote this poem.-->neutral",
            # "Phoebe:Look,  'My vessel so empty with nothing inside.-->neutral",
            # "Phoebe:Now that I've touched you, you seem emptier still.'-->neutral",
            # "Phoebe:He thinks Monica is empty, she is the empty vase!-->negative",
            # "Phoebe:Oh, totally. Oh, God, oh, she seemed so happy too.-->negative",
            # "Joey:Done.-->neutral"

# dialogue = ['Hi Jane , you look great . ', ' You too . Have you lost some weight ? ',
#  ' Yes , I took off 4 kilos . I am glad you notice it . ',
#  ' Not some crazy diet fat I hope . ',
#  ' No no , I just changed my eating habits . I eat a balance meal . and I eat less than before . ',
#  ' Good for you , keep it up . ',
#  "I do not hate diet."]
# dialogue=[
#     "Are you still getting the error?",
# "No, I do not get it now."
# ]


# model_checkpoint = "roberta-base"
model_checkpoint = "finiteautomata/bertweet-base-sentiment-analysis"
# model_checkpoint = "tae898/emoberta-large"
# model_checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Predict emotions for the dialogue
predicted_emotions = predict(dialogue, model, tokenizer, num_past_utterances=0, num_future_utterances=0)
# print(predicted_emotions)
predicted_emotion_labels = map_predictions_to_emotions(predicted_emotions, id2emotion)
print(predicted_emotion_labels)

