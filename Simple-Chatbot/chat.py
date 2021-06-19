import random
import json

import torch

from model import BiLSTM
from data_processing import prepare_test_sentence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
embed_size = data["embed_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
model_state = data["model_state"]
token_to_idx = data["token_to_idx"]
label_to_idx = data["label_to_idx"]

tags = list(label_to_idx.keys())
print(tags)

with open('chat_data.json', 'r') as json_data:
    intents = json.load(json_data)

model = BiLSTM(input_size, embed_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")

while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    X = prepare_test_sentence(sentence, token_to_idx)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    #print(prob.item())
    if prob.item() > 0.25:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
