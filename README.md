# EasyNN - Simple Neural Network Template

A beginner-friendly PyTorch template for creating and training neural networks. Configure your model architecture, training parameters, and data sources all in one place—no deep learning expertise required.

## Features

- **Easy Configuration** – All settings in one section at the top of the file
- **Flexible Architecture** – Customizable hidden layers, activation functions, and dropout
- **Multiple Loss Functions** – Support for classification and regression tasks
- **Multiple Optimizers** – Adam, AdamW, SGD, RMSprop, and more
- **Automatic Data Splitting** – Built-in train/test split functionality
- **Data Scaling** – Optional MinMax scaling for input features
- **Training Visualization** – Automatic loss plot generation
- **Model Evaluation** – Accuracy metrics for classification, MSE/RMSE/MAE/R² for regression

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib

## Installation

### Linux
```bash
# 1. Install Python (if not already installed)
sudo apt install python3 python3-pip python3-venv

# 2. Create and activate virtual environment
python3 -m venv --upgrade-deps venv
source venv/bin/activate

# 3. Install dependencies
pip install torch pandas scikit-learn matplotlib
```

### Windows/macOS
```bash
# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS: source venv/bin/activate

# Install dependencies
pip install torch pandas scikit-learn matplotlib
```

## Quick Start

1. **Prepare your data** – Create a CSV file with your features and target columns

2. **Configure the model** – Edit the USER CONFIGURATION section in `EasyNN.py`:
   ```python
   # Neural Network Architecture
   IN_FEATURES = 63          # Number of input features
   HIDDEN_LAYERS = [128, 64, 32]  # Hidden layer sizes
   OUT_FEATURES = 8          # Number of output classes/values
   
   # Data Settings
   TRAINING_DATA_PATH = 'Train.csv'
   FEATURE_COLUMNS = '0-62'  # Input feature columns
   TARGET_COLUMNS = '63'     # Target/label column
   
   # Training Settings
   EPOCHS = 1500
   LEARNING_RATE = 0.001
   ```

3. **Run training**:
   ```bash
   python EasyNN.py
   ```

4. **Check results** – View `training_loss.png` and console output for metrics

## Configuration Options

### Architecture
| Parameter | Description | Example |
|-----------|-------------|---------|
| `IN_FEATURES` | Number of input features | `63` |
| `HIDDEN_LAYERS` | List of hidden layer sizes | `[128, 64, 32]` |
| `OUT_FEATURES` | Number of outputs | `8` |
| `DROPOUT` | Dropout rate (0 to disable) | `0.2` |
| `ACTIVATION` | Activation function | `'gelu'`, `'relu'`, `'leaky_relu'`, `'elu'`, `'silu'`, `'mish'` |
| `BATCH_NORM` | Enable batch normalization | `True` or `False` |

### Loss Functions
| Value | Use Case |
|-------|----------|
| `'cross_entropy'` | Multi-class classification (default) |
| `'bce_logits'` | Binary classification |
| `'mse'` | Regression |
| `'l1'` | Regression (less sensitive to outliers) |
| `'smooth_l1'` | Regression (balanced) |
| `'huber'` | Regression (robust to outliers) |

### Optimizers
| Value | Description |
|-------|-------------|
| `'adam'` | Adaptive learning rate (default, great for most cases) |
| `'adamw'` | Adam with weight decay (better regularization) |
| `'sgd'` | Stochastic gradient descent (classic) |
| `'rmsprop'` | Good for RNNs |
| `'nadam'` | Adam + Nesterov momentum |
| `'radam'` | Rectified Adam (stable early training) |

### Data Settings
| Parameter | Description |
|-----------|-------------|
| `TRAINING_DATA_PATH` | Path to training CSV file |
| `FEATURE_COLUMNS` | Column indices for features (e.g., `'0-62'` or `'0,1,5-10'`) |
| `TARGET_COLUMNS` | Column indices for targets |
| `USE_TRAINING_DATA_AS_TEST` | Split training data for testing |
| `TRAIN_TEST_SPLIT` | Test set ratio (e.g., `0.2` = 20%) |
| `TEST_DATA_PATH` | Separate test file (if not splitting) |
| `SCALE_DATA` | Normalize features to 0-1 range |

## Output

- **Console** – Training progress, loss values, and evaluation metrics
- **training_loss.png** – Plot of loss over epochs
- **model.pth** – Saved model weights for later use

## Credits

Originally developed following [Codemy.com PyTorch tutorials](https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&si=qN9cA47jvIpXJ6Ez), then extended with additional features for general use.

## License

MIT License - Feel free to use, modify, and distribute.
