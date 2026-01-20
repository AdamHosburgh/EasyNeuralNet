"""
EasyNN - Multilayer Perceptron (MLP) Neural Network Template

A simple, configurable template for building and training feedforward neural
networks using PyTorch. Ideal for classification and regression tasks on
tabular/structured data.

Originally developed following Codemy.com PyTorch tutorials:
https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1

Code assistance provided by Claude (Anthropic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

''' ========== USER CONFIGURATION (Edit these as needed) ========== '''

# Random seed for reproducibility (any integer works)
MANUAL_SEED = 42

# Neural Network Architecture Parameters
IN_FEATURES = 63
HIDDEN_LAYERS = [128, 64, 32, 16]
OUT_FEATURES = 8

# Dropout rate for regularization (0 to disable, 0.1-0.5 typical)
DROPOUT = 0.2

# Activation function ('gelu', 'relu', 'leaky_relu', 'elu', 'silu', 'mish')
ACTIVATION = 'relu'

# Batch Normalization flag
BATCH_NORM = False

# Loss function ('cross_entropy', 'bce', 'bce_logits', 'nll', 'mse', 'l1', 'smooth_l1', 'huber')
LOSS_FUNCTION = 'cross_entropy'

# Training Data File Path
TRAINING_DATA_PATH = 'Train.csv'

# Column indices for features and targets (use ranges like '0-20' or '0,1,5-10')
FEATURE_COLUMNS = '0-62'    # Columns to use as input features
TARGET_COLUMNS = '63'       # Columns to use as targets/labels

# Use some of the training data as test data flag
USE_TRAINING_DATA_AS_TEST = True

# Scale data to between 0 and 1 flag
SCALE_DATA = False

# Train/Test split ratio (only used if USE_TRAINING_DATA_AS_TEST is True)
TRAIN_TEST_SPLIT = 0.15  

# Testing data file path (only used if USE_TRAINING_DATA_AS_TEST is False, Set to = TRAINING_DATA_PATH if unused)
TEST_DATA_PATH = TRAINING_DATA_PATH  

# Optimizer ('adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'nadam', 'radam')
OPTIMIZER = 'adam'

# Learning rate (step size for optimizer)
LEARNING_RATE = 0.01

# Number of training epochs
EPOCHS = 1000

# Path to save the trained model (model.pth recommended)
SAVE_MODEL_FILE_PATH = 'model.pth'

''' ========== MODEL DEFINITION (Do not edit below) ========== '''

class Model(nn.Module):
    """
    Flexible feedforward neural network (MLP) with configurable architecture.

    Supports custom hidden layer sizes, activation functions, dropout,
    and batch normalization.

    Attributes:
        ACTIVATIONS: Dictionary mapping activation names to functions.
    """

    ACTIVATIONS = {
        'gelu': F.gelu,                         # Smooth, modern. Great for transformers and NLP
        'relu': F.relu,                         # Fast, simple default. Can suffer "dying ReLU"
        'leaky_relu': F.leaky_relu,             # Fixes dying ReLU by allowing small negatives
        'elu': F.elu,                           # Smooth, better gradient flow than ReLU
        'silu': F.silu,                         # Self-gated, used in EfficientNet
        'mish': torch.nn.functional.mish,       # Smooth, self-regularizing. Emerging choice
    }

    def __init__(self, in_features=IN_FEATURES, hl=HIDDEN_LAYERS, out_features=OUT_FEATURES,
                 dropout=DROPOUT, activation=ACTIVATION, batch_norm=BATCH_NORM):
        """
        Initialize the neural network.

        Args:
            in_features: Number of input features.
            hl: List of hidden layer sizes.
            out_features: Number of output features/classes.
            dropout: Dropout rate (0 to disable).
            activation: Activation function name.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        # Validate parameters
        if not (0 <= dropout < 1):
            raise ValueError("Dropout must be >=0 and <1")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Activation must be one of {list(self.ACTIVATIONS.keys())}")
        
        # Store configuration
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        
        # Neural network layers
        self.fc = nn.ModuleList()
        # Input layer
        self.fc.append(nn.Linear(in_features, hl[0]))
        # Hidden layers
        for i in range(1, len(hl)):
            self.fc.append(nn.Linear(hl[i-1], hl[i]))
        # Output layer
        self.fc.append(nn.Linear(hl[-1], out_features))

        # Activation function    
        self.activation = self.ACTIVATIONS[activation]
        
        # Batch normalization layers
        if self.batch_norm:
            self.bn = nn.ModuleList()
            for i in range(len(hl)):
                self.bn.append(nn.BatchNorm1d(hl[i]))

        # Dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            if self.batch_norm:
                x = self.bn[i](x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)

        x = self.fc[-1](x)
        return x
    
torch.manual_seed(MANUAL_SEED)
model = Model()

''' ========== DATA LOADING AND PREPROCESSING ========== '''

# Available loss functions
LOSS_FUNCTIONS = {
    # Classification
    'cross_entropy': nn.CrossEntropyLoss,       # Multi-class classification (raw logits)
    'bce': nn.BCELoss,                          # Binary classification (requires sigmoid output)
    'bce_logits': nn.BCEWithLogitsLoss,         # Binary classification (raw logits, more stable)
    'nll': nn.NLLLoss,                          # Negative log likelihood (requires log_softmax output)
    
    # Regression
    'mse': nn.MSELoss,                          # Mean Squared Error
    'l1': nn.L1Loss,                            # Mean Absolute Error
    'smooth_l1': nn.SmoothL1Loss,               # Smooth L1 (less sensitive to outliers)
    'huber': nn.HuberLoss,                      # Huber loss (similar to smooth_l1)
}

# Validate loss function choice
if LOSS_FUNCTION not in LOSS_FUNCTIONS:
    raise ValueError(f"Loss function must be one of {list(LOSS_FUNCTIONS.keys())}")

# Available optimizers
OPTIMIZERS = {
    'adam': torch.optim.Adam,           # Adaptive learning rate. Great default choice
    'adamw': torch.optim.AdamW,         # Adam + weight decay. Best for regularization
    'sgd': torch.optim.SGD,             # Classic gradient descent. Good with momentum
    'rmsprop': torch.optim.RMSprop,     # Adaptive learning rate. Good for RNNs
    'adagrad': torch.optim.Adagrad,     # Adapts per-parameter. Good for sparse data
    'nadam': torch.optim.NAdam,         # Adam + Nesterov momentum. Faster convergence
    'radam': torch.optim.RAdam,         # Rectified Adam. More stable early training
}

# Validate optimizer choice
if OPTIMIZER not in OPTIMIZERS:
    raise ValueError(f"Optimizer must be one of {list(OPTIMIZERS.keys())}")

# Load training data
Training_df = pd.read_csv(TRAINING_DATA_PATH)

if USE_TRAINING_DATA_AS_TEST:
    train_df, test_df = train_test_split(Training_df, test_size=TRAIN_TEST_SPLIT, random_state=MANUAL_SEED, shuffle=True)
else:
    train_df = Training_df
    test_df = pd.read_csv(TEST_DATA_PATH)

def parse_column_range(range_str):
    """
    Parse column range string into list of indices.

    Supports formats: '0-20', '5', '0,1,5-10', '0-5,10,15-20'

    Args:
        range_str: String specifying column indices or ranges.

    Returns:
        List of integer column indices.
    """
    indices = []
    parts = range_str.replace(' ', '').split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return indices

# Parse column selections
feature_cols = parse_column_range(FEATURE_COLUMNS)
target_cols = parse_column_range(TARGET_COLUMNS)

# Extract features and targets using column indices
X_train = train_df.iloc[:, feature_cols].values
y_train = train_df.iloc[:, target_cols].values

X_test = test_df.iloc[:, feature_cols].values
y_test = test_df.iloc[:, target_cols].values

# Scale feature data if enabled (scales to 0-1 range)
if SCALE_DATA:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Flatten targets if single column (for classification)
if len(target_cols) == 1:
    y_train = y_train.ravel()
    y_test = y_test.ravel()

# Print dataset sizes
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set: {X_test.shape[0]} samples")

# Convert X features to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y targets to appropriate tensor types
if LOSS_FUNCTION in ['cross_entropy', 'nll']:
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
else:
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

# Initialize loss criterion and optimizer
loss_criterion = LOSS_FUNCTIONS[LOSS_FUNCTION]()
optimizer = OPTIMIZERS[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

''' ========== MODEL TRAINING ========== '''

epochs = EPOCHS
losses = []

for i in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train)
    loss = loss_criterion(y_pred, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item():.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig('training_loss.png')
print("Training loss plot saved as 'training_loss.png'")

# Evaluate model on test set
with torch.no_grad():
    y_eval = model.forward(X_test)
    test_loss = loss_criterion(y_eval, y_test)

print(f"Test Loss: {test_loss.item():.4f}")

# Classification metrics
if LOSS_FUNCTION in ['cross_entropy', 'nll']:
    model.eval()
    with torch.no_grad():
        y_pred_classes = torch.argmax(model(X_test), dim=1)
        correct = (y_pred_classes == y_test).sum().item()
        accuracy = correct / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{y_test.size(0)} correct)")

    print("\nFirst 20 Predictions vs Actual:")
    for i in range(min(20, y_test.size(0))):
        print(f"{i + 1}.) Predicted: {y_pred_classes[i].item()} \t Actual: {y_test[i].item()}")

# Regression metrics
else:
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

        mse = torch.mean((y_pred - y_test) ** 2).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
        mae = torch.mean(torch.abs(y_pred - y_test)).item()

        ss_res = torch.sum((y_test - y_pred) ** 2)
        ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

    print("\n--- Regression Metrics ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\nFirst 20 Predictions vs Actual:")
    for i in range(min(20, y_test.size(0))):
        if len(target_cols) == 1:
            print(f"{i + 1}.) Predicted: {y_pred[i].item():.4f} \t Actual: {y_test[i].item():.4f}")
        else:
            pred_str = ', '.join([f"{v:.2f}" for v in y_pred[i].tolist()])
            actual_str = ', '.join([f"{v:.2f}" for v in y_test[i].tolist()])
            print(f"{i + 1}.) Predicted: [{pred_str}] \t Actual: [{actual_str}]")

# Save model
torch.save(model.state_dict(), SAVE_MODEL_FILE_PATH)
print(f"Model saved as '{SAVE_MODEL_FILE_PATH}'")    