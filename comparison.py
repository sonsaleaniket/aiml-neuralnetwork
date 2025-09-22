"""
Neural Network Framework Comparison: TensorFlow vs PyTorch
=========================================================

This script compares TensorFlow and PyTorch implementations of the same CNN architecture
on the CIFAR-10 dataset, measuring performance, training time, and accuracy.

Author: AI Assistant
Date: 2024
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class TensorFlowCNN:
    """TensorFlow implementation of CNN for CIFAR-10 classification."""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the CNN model architecture."""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def load_data(self):
        """Load and preprocess CIFAR-10 dataset."""
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def train(self, train_data, test_data, epochs=10, batch_size=32):
        """Train the TensorFlow model."""
        (train_images, train_labels) = train_data
        (test_images, test_labels) = test_data
        
        start_time = time.time()
        
        self.history = self.model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_images, test_labels),
            verbose=1
        )
        
        training_time = time.time() - start_time
        return training_time
    
    def evaluate(self, test_data):
        """Evaluate the TensorFlow model."""
        (test_images, test_labels) = test_data
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        return test_loss, test_accuracy


class PyTorchCNN(nn.Module):
    """PyTorch implementation of CNN for CIFAR-10 classification."""
    
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate the correct input size for fc1
        # After conv1 + pool: 32x32 -> 16x16
        # After conv2 + pool: 16x16 -> 8x8  
        # After conv3: 8x8
        # So fc1 input should be 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten with proper batch size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PyTorchTrainer:
    """PyTorch training and evaluation class."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def load_data(self, batch_size=32):
        """Load and preprocess CIFAR-10 dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        return trainloader, testloader
    
    def train(self, trainloader, epochs=10, learning_rate=0.001):
        """Train the PyTorch model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = running_loss / len(trainloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        training_time = time.time() - start_time
        return training_time
    
    def evaluate(self, testloader):
        """Evaluate the PyTorch model."""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(testloader)
        return avg_loss, accuracy


class FrameworkComparison:
    """Main class for comparing TensorFlow and PyTorch frameworks."""
    
    def __init__(self):
        self.tf_cnn = TensorFlowCNN()
        self.pytorch_cnn = PyTorchCNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_trainer = PyTorchTrainer(self.pytorch_cnn, self.device)
        
    def run_comparison(self, epochs=10, batch_size=32):
        """Run the complete comparison between frameworks."""
        print("=" * 60)
        print("Neural Network Framework Comparison: TensorFlow vs PyTorch")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        print("=" * 60)
        
        # TensorFlow Implementation
        print("\n🔵 TensorFlow Implementation")
        print("-" * 30)
        
        # Build and load data
        self.tf_cnn.build_model()
        train_data, test_data = self.tf_cnn.load_data()
        
        # Train TensorFlow model
        tf_training_time = self.tf_cnn.train(train_data, test_data, epochs, batch_size)
        tf_loss, tf_accuracy = self.tf_cnn.evaluate(test_data)
        
        print(f"TensorFlow Training Time: {tf_training_time:.2f} seconds")
        print(f"TensorFlow Test Accuracy: {tf_accuracy:.4f}")
        print(f"TensorFlow Test Loss: {tf_loss:.4f}")
        
        # PyTorch Implementation
        print("\n🟠 PyTorch Implementation")
        print("-" * 30)
        
        # Load data
        trainloader, testloader = self.pytorch_trainer.load_data(batch_size)
        
        # Train PyTorch model
        pytorch_training_time = self.pytorch_trainer.train(trainloader, epochs)
        pytorch_loss, pytorch_accuracy = self.pytorch_trainer.evaluate(testloader)
        
        print(f"PyTorch Training Time: {pytorch_training_time:.2f} seconds")
        print(f"PyTorch Test Accuracy: {pytorch_accuracy:.2f}%")
        print(f"PyTorch Test Loss: {pytorch_loss:.4f}")
        
        # Comparison Results
        print("\n📊 Comparison Results")
        print("=" * 30)
        print(f"Accuracy Difference: {abs(tf_accuracy - pytorch_accuracy/100):.4f}")
        print(f"Training Time Difference: {abs(tf_training_time - pytorch_training_time):.2f} seconds")
        
        if tf_training_time < pytorch_training_time:
            print("🏆 TensorFlow is faster for training")
        else:
            print("🏆 PyTorch is faster for training")
            
        if tf_accuracy > pytorch_accuracy/100:
            print("🏆 TensorFlow has higher accuracy")
        else:
            print("🏆 PyTorch has higher accuracy")
        
        return {
            'tensorflow': {
                'training_time': tf_training_time,
                'accuracy': tf_accuracy,
                'loss': tf_loss
            },
            'pytorch': {
                'training_time': pytorch_training_time,
                'accuracy': pytorch_accuracy/100,
                'loss': pytorch_loss
            }
        }


def main():
    """Main function to run the comparison."""
    try:
        comparison = FrameworkComparison()
        results = comparison.run_comparison(epochs=5, batch_size=32)  # Reduced epochs for demo
        
        print("\n✅ Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()