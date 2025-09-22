# AI/ML Neural Network Project

This project implements neural networks for image classification using both PyTorch and TensorFlow frameworks. The project includes individual implementations and a comprehensive comparison tool to evaluate both frameworks' performance on the same task.

## 🚀 Project Structure

```
├── neural_network_pytorch.py     # PyTorch neural network implementation (MNIST + Azure ML)
├── neural_network_tensorflow.py  # TensorFlow neural network implementation (Fashion MNIST)
├── comparison.py                 # Framework comparison tool (TensorFlow vs PyTorch)
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation file
├── .venv/                        # Virtual environment directory
├── .azureml/                     # Azure ML workspace configuration
├── simple_nn.pth                 # Trained PyTorch model file (generated after training)
└── data/                         # Dataset directory (created automatically)
```

## 📦 Dependencies

- **azureml-core**: Azure Machine Learning core library
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities for PyTorch
- **tensorflow**: TensorFlow deep learning framework
- **numpy**: Numerical computing library
- **matplotlib**: Plotting and visualization library

## 🛠️ Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv .venv
```

### 2. Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Azure ML (for PyTorch version)
- Update `subscription_id`, `resource_group`, and `location` in `neural_network_pytorch.py`
- Ensure you have proper Azure credentials configured

## 🏃‍♂️ Running the Project

### Individual Implementations

#### PyTorch Neural Network (MNIST + Azure ML)
```bash
python neural_network_pytorch.py
```

#### TensorFlow Neural Network (Fashion MNIST)
```bash
python neural_network_tensorflow.py
```

### Framework Comparison Tool

#### Run Complete Comparison
```bash
python comparison.py
```

The comparison tool will:
- Train identical CNN architectures on CIFAR-10 dataset
- Measure training time, accuracy, and loss for both frameworks
- Provide detailed performance comparison
- Display results in a formatted, easy-to-read format

## 🧠 Neural Network Architectures

### PyTorch Version (MNIST)
- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 0.01

### TensorFlow Version (Fashion MNIST)
- **Input Layer**: Flatten layer (28×28 → 784)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (clothing categories 0-9)
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam optimizer

### Comparison Tool (CIFAR-10)
Both frameworks use identical CNN architecture:
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 3**: 64 filters, 3×3 kernel, ReLU activation
- **Dense Layer**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (CIFAR-10 classes)

## 📊 Training Results

### PyTorch Version (MNIST)
- Epoch 1: ~81% accuracy
- Epoch 2: ~89% accuracy
- Epoch 3: ~90% accuracy
- Epoch 4: ~91% accuracy
- Epoch 5: ~92% accuracy

### TensorFlow Version (Fashion MNIST)
- Epoch 1: ~82% accuracy
- Epoch 2: ~86% accuracy
- Epoch 3: ~87% accuracy
- Epoch 4: ~88% accuracy
- Epoch 5: ~89% accuracy
- Final Test Accuracy: ~88%

### Framework Comparison Results
Based on recent test runs:
- **TensorFlow**: 69.13% accuracy, 80.19 seconds training time
- **PyTorch**: 69.23% accuracy, 222.47 seconds training time
- **Winner**: TensorFlow for speed, PyTorch for accuracy (minimal difference)

## ✨ Features

### PyTorch Version
- Error handling for Azure ML operations
- Progress tracking with loss and accuracy metrics
- Automatic model saving and Azure ML registration
- Suppressed deprecation warnings for cleaner output
- Robust dataset loading with error handling

### TensorFlow Version
- Simple and clean implementation
- Automatic dataset downloading
- Real-time training progress display
- Built-in model evaluation
- No external dependencies (pure TensorFlow)

### Comparison Tool
- **Fair Comparison**: Identical architectures and datasets
- **Performance Metrics**: Training time, accuracy, and loss measurement
- **Device Detection**: Automatic CPU/GPU detection
- **Error Handling**: Robust error handling and recovery
- **Clear Output**: Formatted results with progress indicators
- **Modular Design**: Easy to extend and customize

## 🔧 Azure ML Integration

- **Workspace**: your-workspace-name
- **Subscription**: your-subscription-id
- **Resource Group**: your-resource-group
- **Location**: your-location
- **Model Name**: simple_nn

## 🐛 Troubleshooting

### 1. Azure Authentication Issues (PyTorch version)
- Ensure you're logged in to Azure CLI: `az login`
- Check workspace configuration parameters

### 2. Dataset Download Issues
- Ensure internet connection for MNIST/Fashion MNIST/CIFAR-10 download
- Check disk space for dataset storage

### 3. Model Registration Issues (PyTorch version)
- Verify Azure ML workspace permissions
- Check if model file exists before registration

### 4. TensorFlow Import Issues
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check Python version compatibility (3.8+)

### 5. Virtual Environment Issues
- Ensure Python 3.8+ is installed
- Recreate virtual environment if corrupted

### 6. Comparison Tool Issues
- Ensure both TensorFlow and PyTorch are properly installed
- Check that all dependencies are up to date
- Verify sufficient system memory for CIFAR-10 dataset

## 🚀 Future Enhancements

- [ ] Add validation dataset evaluation for all versions
- [ ] Implement model deployment as web service (PyTorch)
- [ ] Add hyperparameter tuning capabilities
- [ ] Include model visualization and architecture diagrams
- [ ] Add support for custom datasets
- [ ] Implement advanced CNN architectures (ResNet, VGG, etc.)
- [ ] Create unified training pipeline
- [ ] Add GPU acceleration comparison
- [ ] Implement distributed training comparison
- [ ] Add memory usage profiling

## 📈 Performance Comparison

The comparison tool provides detailed insights into:

- **Training Speed**: Which framework trains faster
- **Memory Usage**: Resource consumption comparison
- **Accuracy**: Model performance on test data
- **Ease of Use**: Code complexity and readability
- **Ecosystem**: Available tools and libraries

## 📞 Contact

For questions or issues, please check the:
- [Azure ML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [TensorFlow documentation](https://www.tensorflow.org/learn)

---

**Last Updated**: January 2025  
**Version**: 2.0 (Added Framework Comparison Tool)
