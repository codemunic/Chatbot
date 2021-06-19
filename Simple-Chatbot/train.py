from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn

from chat_dataset import ChatDataset
from model import BiLSTM

dataset = ChatDataset("chat_data.json")

#hyperparameter
batch_size = 8
num_epochs = 500
input_size = dataset.vocab_size
embedding_size = 128
hidden_size = 64
output_size = dataset.class_size
dp = 0.2
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

print("dataset_size: ", len(dataset))
print("dataloader_size: ", len(train_loader))



model = BiLSTM(input_size, embedding_size, hidden_size, output_size, dp).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        #print("words: ", words)
        #print("labels: ", labels)
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"embed_size": embedding_size,
"hidden_size": hidden_size,
"output_size": output_size,
"token_to_idx": dataset.token_to_idx,
"label_to_idx": dataset.label_to_idx
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')