import torch
import torch.nn as nn
import torch.optim as optim

# XOR data
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Perceptron model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),  # Změna aktivační funkce
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

# Initializer
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Supervisor class
class Supervisor:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.previous_loss = float('inf')

    def train_step(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Model, loss, optimizer
model = XORModel()
model.apply(init_weights)  # Použití inicializace
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
supervisor = Supervisor(model, optimizer, criterion)

# Training loop
for epoch in range(2000):
    loss = supervisor.train_step(inputs, labels)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/2000], Loss: {loss:.4f}')
    if loss < 0.01:
        print(f"Training stopped early at epoch {epoch + 1}")
        break

# Test
print("Testing:")
with torch.no_grad():
    predictions = model(inputs)
    print("Inputs: ", inputs)
    print("Predictions: ", predictions)
