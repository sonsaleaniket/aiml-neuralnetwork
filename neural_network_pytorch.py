from azureml.core import Workspace, Model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import warnings
import os

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="azureml")

# Create or retrieve an existing Azure ML workspace
try:
    ws = Workspace.get(name='your-workspace-name',
                          subscription_id='your-subscription-id',
                          resource_group='your-resource-group',
                          location='your-location')
    print("Successfully connected to existing Azure ML workspace")
except Exception as e:
    print(f"Error connecting to workspace: {e}")
    print("Please check your Azure credentials and workspace configuration")
    exit(1)

# Write configuration to the workspace config file
try:
    ws.write_config(path='.azureml')
    print("Workspace configuration saved successfully")
except Exception as e:
    print(f"Error saving workspace configuration: {e}")

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (784 input features)
        self.fc2 = nn.Linear(128, 10)   # Output layer (10 classes)
        self.relu = nn.ReLU()           # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = self.fc2(x)             # Output layer
        return x

# Initialize the neural network
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Load MNIST dataset
print("Loading MNIST dataset...")
try:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=32, shuffle=True)
    print("MNIST dataset loaded successfully")
except Exception as e:
    print(f"Error loading MNIST dataset: {e}")
    exit(1)

# Train the model
print("Starting model training...")
num_epochs = 5
model.train()  # Set model to training mode

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(inputs.view(-1, 784))  # Flatten input
        loss = criterion(outputs, labels)      # Compute loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Training completed successfully!")

# Save the trained model
try:
    torch.save(model.state_dict(), 'simple_nn.pth')
    print("Model saved successfully as 'simple_nn.pth'")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# Register the model in Azure
try:
    registered_model = Model.register(workspace=ws, model_path='simple_nn.pth', model_name='simple_nn')
    print(f"Model registered successfully in Azure ML: {registered_model.name}")
except Exception as e:
    print(f"Error registering model in Azure ML: {e}")
    print("Model saved locally but not registered in Azure ML")

# Deploying as a web service requires more steps such as creating a scoring script and environment.