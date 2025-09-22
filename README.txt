AI/ML Neural Network Project
=============================

This project implements a simple neural network for digit classification using the MNIST dataset, 
integrated with Azure Machine Learning services.

PROJECT STRUCTURE
=================
- neural_network.py    : Main neural network implementation
- requirements.txt     : Python dependencies
- README.txt          : This documentation file
- .venv/              : Virtual environment directory
- .azureml/           : Azure ML workspace configuration
- simple_nn.pth       : Trained model file (generated after training)
- data/               : MNIST dataset directory (created automatically)

DEPENDENCIES
============
- azureml-core        : Azure Machine Learning core library
- torch               : PyTorch deep learning framework
- torchvision         : Computer vision utilities for PyTorch
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

2. Run the neural network:
   python neural_network.py

The script will:
- Connect to Azure ML workspace
- Download MNIST dataset (if not already present)
- Train a neural network for 5 epochs
- Display training progress with loss and accuracy
- Save the trained model locally
- Register the model in Azure ML

NEURAL NETWORK ARCHITECTURE
===========================
- Input Layer: 784 neurons (28x28 flattened MNIST images)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9)
- Loss Function: Cross Entropy Loss
- Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01

TRAINING RESULTS
================
The model typically achieves:
- Epoch 1: ~81% accuracy
- Epoch 2: ~89% accuracy
- Epoch 3: ~90% accuracy
- Epoch 4: ~91% accuracy
- Epoch 5: ~92% accuracy

FEATURES
========
- Error handling for Azure ML operations
- Progress tracking with loss and accuracy metrics
- Automatic model saving and Azure ML registration
- Suppressed deprecation warnings for cleaner output
- Robust dataset loading with error handling


AZURE ML INTEGRATION
====================
- Workspace: your-workspace-name
- Subscription: your-subscription-id
- Resource Group: your-resource-group
- Location: your-location
- Model Name: simple_nn

TROUBLESHOOTING
===============
1. Azure Authentication Issues:
   - Ensure you're logged in to Azure CLI: az login
   - Check workspace configuration parameters

2. Dataset Download Issues:
   - Ensure internet connection for MNIST download
   - Check disk space for dataset storage

3. Model Registration Issues:
   - Verify Azure ML workspace permissions
   - Check if model file exists before registration

4. Virtual Environment Issues:
   - Ensure Python 3.8+ is installed
   - Recreate virtual environment if corrupted

FUTURE ENHANCEMENTS
===================
- Add validation dataset evaluation
- Implement model deployment as web service
- Add hyperparameter tuning
- Include model visualization
- Add support for custom datasets

CONTACT
=======
For questions or issues, please check the Azure ML documentation or 
PyTorch tutorials for additional guidance.

Last Updated: January 2025
