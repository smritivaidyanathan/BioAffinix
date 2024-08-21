import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def createData():
    smiles_npz = np.load("Data_Base_Building/smiles.npz")
    df = pd.read_csv('Data_Base_Building/drug_target_interactions_IC50.csv')
    allsmiles = df["SMILES"]

    max_shape = 0
    for smiles in smiles_npz:
        max_shape = max(max_shape, (smiles_npz[smiles].shape)[0])

    X = []
    for smiles in allsmiles:
        embedding = smiles_npz[smiles]
        shape_embedding = max_shape - embedding.shape[0]
        padded_embedding = np.pad(embedding, [(0, shape_embedding), (0, 0)], mode='constant')
        X.append(padded_embedding)

    X = np.array(X)
    y = np.array(df["Affinity"])


    
    nan_indices = np.isnan(y)
    X = X[~nan_indices]
    y = y[~nan_indices]

    y_min = np.min(y)
    y_max = np.max(y)

    print (f"Range before normalizing:{y_max-y_min}. Avg value: {np.mean(y)}")

    alpha = 100

    y = alpha * (y - y_min) / (y_max - y_min)
    y = y.astype(np.float32)
    print (f"Range before normalizing:{np.max(y)-np.min(y)}. Avg value: {np.mean(y)}")

    print(X.shape)
    print(y.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class ELEVENBETA(nn.Module):
    def __init__(self):
        super(ELEVENBETA, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(64 * 48 * 128, 256)
        self.fc2 = nn.Linear(256, 1) 
        
    def forward(self, x):
        m = nn.ReLU()
        x = self.conv1(x)
        x = m(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = m(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 48 * 128) 
        x = m(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
X_train, X_test, y_train, y_test = createData()
print(X_train.dtype)
print(y_train)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_train_tensor = X_train_tensor.unsqueeze(1)
print(torch.isnan(X_train_tensor).any())
print(torch.isinf(X_train_tensor).any())
print(torch.isnan(y_train_tensor).any())
print(torch.isinf(y_train_tensor).any())

print ("Tensor's created")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = ELEVENBETA()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print ("beginning epochs")
epsilon = 0.1  # Define the tolerance range

for epoch in range(20):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in tqdm(train_dataloader, desc=f"Running Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += ((outputs - labels).abs() < epsilon).sum().item()
        total_predictions += labels.size(0)
    epoch_accuracy = (correct_predictions / total_predictions) * 50
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader)}, Accuracy: {epoch_accuracy:.2f}%')

print('Finished Training')

torch.save(model.state_dict(), 'Data_Base_Building/smiles_cnn_model.pth')


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_test_tensor = X_test_tensor.unsqueeze(1)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += ((outputs - labels).abs() < epsilon).sum().item()
        total_predictions += labels.size(0)
        
    test_accuracy = (correct_predictions / total_predictions) * 100

    print(f'Test Loss: {test_loss/len(test_dataloader)}, Test accuracy: {test_accuracy}')


