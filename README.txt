AI/ML Neural Network Project
=============================

This project implements neural networks for image classification using both PyTorch and TensorFlow frameworks.
The PyTorch version uses MNIST dataset with Azure Machine Learning integration, while the TensorFlow version 
uses Fashion MNIST dataset for clothing classification.

PROJECT STRUCTURE
=================
- neural_network.py         : PyTorch neural network implementation (MNIST + Azure ML)
- neural_network_tensorflow.py : TensorFlow neural network implementation (Fashion MNIST)
- requirements.txt          : Python dependencies
- README.txt               : This documentation file
- .venv/                   : Virtual environment directory
- .azureml/                : Azure ML workspace configuration
- simple_nn.pth            : Trained PyTorch model file (generated after training)
- data/                    : Dataset directory (created automatically)

DEPENDENCIES
============
- azureml-core        : Azure Machine Learning core library
- torch               : PyTorch deep learning framework
- torchvision         : Computer vision utilities for PyTorch
- tensorflow          : TensorFlow deep learning framework
- numpy               : Numerical computing library
- pillow              : Image processing library

SETUP INSTRUCTIONS
==================

1. Create Virtual Environment:
   python -m venv .venv

2. Activate Virtual Environment:
   Windows: .venv\Scripts\activate
   Linux/Mac: source .venv/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Configure Azure ML:
   - Update subscription_id, resource_group, and location in neural_network.py
   - Ensure you have proper Azure credentials configured


RUNNING THE PROJECT
===================

1. Activate the virtual environment:
   .venv\Scripts\activate

2. Run PyTorch neural network (MNIST + Azure ML):
   python neural_network.py

3. Run TensorFlow neural network (Fashion MNIST):
   python neural_network_tensorflow.py

PYTORCH VERSION (neural_network.py):
- Connects to Azure ML workspace
- Downloads MNIST dataset (if not already present)
- Trains a neural network for 5 epochs
- Displays training progress with loss and accuracy
- Saves the trained model locally
- Registers the model in Azure ML

TENSORFLOW VERSION (neural_network_tensorflow.py):
- Downloads Fashion MNIST dataset automatically
- Trains a neural network for 10 epochs
- Displays training progress with loss and accuracy
- Evaluates model on test set
- Achieves ~88% test accuracy

NEURAL NETWORK ARCHITECTURES
============================

PYTORCH VERSION (MNIST):
- Input Layer: 784 neurons (28x28 flattened MNIST images)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9)
- Loss Function: Cross Entropy Loss
- Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01

TENSORFLOW VERSION (Fashion MNIST):
- Input Layer: Flatten layer (28x28 → 784)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (clothing categories 0-9)
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam optimizer

TRAINING RESULTS
================

PYTORCH VERSION (MNIST):
- Epoch 1: ~81% accuracy
- Epoch 2: ~89% accuracy
- Epoch 3: ~90% accuracy
- Epoch 4: ~91% accuracy
- Epoch 5: ~92% accuracy

TENSORFLOW VERSION (Fashion MNIST):
- Epoch 1: ~82% accuracy
- Epoch 2: ~86% accuracy
- Epoch 3: ~87% accuracy
- Epoch 4: ~88% accuracy
- Epoch 5: ~89% accuracy
- Epoch 6: ~89% accuracy
- Epoch 7: ~90% accuracy
- Epoch 8: ~90% accuracy
- Epoch 9: ~90% accuracy
- Epoch 10: ~91% accuracy
- Final Test Accuracy: ~88%

FEATURES
========

PYTORCH VERSION:
- Error handling for Azure ML operations
- Progress tracking with loss and accuracy metrics
- Automatic model saving and Azure ML registration
- Suppressed deprecation warnings for cleaner output
- Robust dataset loading with error handling

TENSORFLOW VERSION:
- Simple and clean implementation
- Automatic dataset downloading
- Real-time training progress display
- Built-in model evaluation
- No external dependencies (pure TensorFlow)


AZURE ML INTEGRATION
====================
- Workspace: your-workspace-name
- Subscription: your-subscription-id
- Resource Group: your-resource-group
- Location: your-location
- Model Name: simple_nn

TROUBLESHOOTING
===============
1. Azure Authentication Issues (PyTorch version):
   - Ensure you're logged in to Azure CLI: az login
   - Check workspace configuration parameters

2. Dataset Download Issues:
   - Ensure internet connection for MNIST/Fashion MNIST download
   - Check disk space for dataset storage

3. Model Registration Issues (PyTorch version):
   - Verify Azure ML workspace permissions
   - Check if model file exists before registration

4. TensorFlow Import Issues:
   - Ensure TensorFlow is installed: pip install tensorflow
   - Check Python version compatibility (3.8+)

5. Virtual Environment Issues:
   - Ensure Python 3.8+ is installed
   - Recreate virtual environment if corrupted

FUTURE ENHANCEMENTS
===================
- Add validation dataset evaluation for both versions
- Implement model deployment as web service (PyTorch)
- Add hyperparameter tuning
- Include model visualization
- Add support for custom datasets
- Implement CNN architectures for better performance
- Add model comparison between PyTorch and TensorFlow
- Create unified training pipeline

CONTACT
=======
For questions or issues, please check the Azure ML documentation, 
PyTorch tutorials, or TensorFlow documentation for additional guidance.

Last Updated: January 2025
