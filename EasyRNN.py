"""
EasyRNN - Recurrent Neural Network Template

A simple, configurable template for building and training Recurrent Neural
Networks using PyTorch. Ideal for time series forecasting, sequence prediction,
and any task where the order of data matters.

Use cases:
    - Time series forecasting (stock prices, sensor data, weather)
    - Natural language processing (text classification, sentiment analysis)
    - Predictive maintenance (RUL prediction, anomaly detection)

Code assistance provided by Claude (Anthropic).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

''' ========== USER CONFIGURATION (Edit these as needed) ========== '''

# Random seed for reproducibility (any integer works)
MANUAL_SEED = 42

# ===== RNN Architecture Parameters =====
INPUT_SIZE = 24         # Number of features per timestep
HIDDEN_SIZE = 128       # Number of neurons in hidden layers
NUM_LAYERS = 2          # Number of stacked RNN layers
OUTPUT_SIZE = 1         # Number of output values (1 for regression, N for classification)

# RNN cell type: 'lstm' (best for long sequences), 'gru' (faster), 'rnn' (basic)
RNN_TYPE = 'lstm'

# Bidirectional RNN flag (processes sequence forwards and backwards)
BIDIRECTIONAL = False

# Dropout rate for regularization (0 to disable, 0.1-0.5 typical)
# Note: Dropout only applies between RNN layers when NUM_LAYERS > 1
DROPOUT = 0

# ===== Sequence Parameters =====
SEQUENCE_LENGTH = 50    # Number of timesteps per sequence (lookback window)

# Sequence ID column index (e.g., engine_id, patient_id, sensor_id)
# Set to None if data is a single continuous sequence
SEQUENCE_ID_COLUMN = 0  # Column index for sequence identifier

# Time/order column index (e.g., cycle, timestamp, time)
# Set to None if data is already sorted
TIME_COLUMN = 1         # Column index for time/order information

# ===== Loss Function =====
# Classification: 'cross_entropy', 'bce', 'bce_logits', 'nll'
# Regression: 'mse', 'l1', 'smooth_l1', 'huber'
LOSS_FUNCTION = 'mse'

# ===== Data Configuration =====
TRAINING_DATA_PATH = 'Train1.csv'

# Column indices for features and targets (e.g., '0-20' or '0,1,5-10')
# Or use column names: 'temp,pressure,vibration'
FEATURE_COLUMNS = '2-25'  # Columns to use as input features
TARGET_COLUMNS = '26'     # Column(s) to use as targets/labels

# Use some of the training data as test data flag
USE_TRAINING_DATA_AS_TEST = True

# Scale feature data to 0-1 range (recommended for RNNs)
SCALE_DATA = True

# Train/Test split ratio (only used if USE_TRAINING_DATA_AS_TEST is True)
TRAIN_TEST_SPLIT = 0.2

# Testing data file path (only used if USE_TRAINING_DATA_AS_TEST is False)
TEST_DATA_PATH = TRAINING_DATA_PATH

# ===== Training Parameters =====
OPTIMIZER = 'adam'      # 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'nadam', 'radam'
LEARNING_RATE = 0.001   # Step size for optimizer (0.001 is good default for RNNs)
BATCH_SIZE = 32         # Number of sequences per training batch
EPOCHS = 100            # Number of training epochs

# Path to save the trained model
SAVE_MODEL_FILE_PATH = 'rnn_model.pth'

''' ========== MODEL DEFINITION (Do not edit below) ========== '''

# Device configuration (auto-detects GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    """
    Flexible RNN model supporting LSTM, GRU, and vanilla RNN architectures.
    
    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, output_size)
    """
    
    # Available RNN cell types
    RNN_TYPES = {
        'lstm': nn.LSTM,    # Long Short-Term Memory: best for long sequences, handles vanishing gradients
        'gru': nn.GRU,      # Gated Recurrent Unit: simpler than LSTM, often similar performance
        'rnn': nn.RNN,      # Vanilla RNN: simple but suffers from vanishing gradients
    }

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                 num_layers=NUM_LAYERS, rnn_type=RNN_TYPE, dropout=DROPOUT, bidirectional=BIDIRECTIONAL):
        """
        Initialize the RNN model.

        Args:
            input_size: Number of features per timestep.
            hidden_size: Number of neurons in hidden layers.
            output_size: Number of output values.
            num_layers: Number of stacked RNN layers.
            rnn_type: Type of RNN cell ('lstm', 'gru', 'rnn').
            dropout: Dropout rate for regularization.
            bidirectional: Whether to use bidirectional RNN.
        """
        super().__init__()

        # Validate parameters
        if rnn_type not in self.RNN_TYPES:
            raise ValueError(f"RNN type must be one of {list(self.RNN_TYPES.keys())}")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout must be >=0 and <1")
        
        # Store configuration
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create RNN layer
        rnn_dropout = dropout if num_layers > 1 else 0  # Dropout only between layers
        self.rnn = self.RNN_TYPES[rnn_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Input shape: (batch, seq, features)
            dropout=rnn_dropout,
            bidirectional=bidirectional
        )
        
        # Dropout layer for output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Fully connected output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        """
        Forward pass through the RNN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)

        # Initialize hidden state(s)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        # Take output from the last timestep
        out = out[:, -1, :]
        
        # Apply dropout if enabled
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Pass through fully connected layer
        out = self.fc(out)
        return out

# Set random seed for reproducibility
torch.manual_seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)

# Print device info
print(f"Using device: {DEVICE}")

# Initialize model and move to device
model = RNNModel().to(DEVICE)
print(f"\nModel Architecture: {RNN_TYPE.upper()}")
print(f"  Input size: {INPUT_SIZE}")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Num layers: {NUM_LAYERS}")
print(f"  Output size: {OUTPUT_SIZE}")
print(f"  Bidirectional: {BIDIRECTIONAL}")
print(f"  Dropout: {DROPOUT}")

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

def parse_column_range(range_str, df=None):
    """
    Parse column range string into list of indices or column names.

    Supports formats: '0-20', '5', '0,1,5-10', '0-5,10,15-20'
    Also supports column names: 'temp,pressure,vibration'

    Args:
        range_str: String specifying column indices, ranges, or names.
        df: Optional DataFrame (unused, for API compatibility).

    Returns:
        List of integer indices or column name strings.
    """
    # Check if it's column names (contains letters)
    if any(c.isalpha() for c in range_str):
        return [col.strip() for col in range_str.split(',')]
    
    # Parse as indices
    indices = []
    parts = range_str.replace(' ', '').split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return indices

def create_sequences(df, feature_cols, target_cols, sequence_length,
                     sequence_id_col=None, time_col=None):
    """
    Create sequences from DataFrame for RNN training.

    For data with multiple sequences (e.g., multiple engines/sensors),
    groups by sequence_id_col and creates sliding window sequences within
    each group. For single continuous sequence, creates sliding window
    over entire dataset.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column indices or names.
        target_cols: List of target column indices or names.
        sequence_length: Number of timesteps per sequence.
        sequence_id_col: Column index for sequence identifier (None for single sequence).
        time_col: Column index for time ordering (None if pre-sorted).

    Returns:
        Tuple of (X_sequences, y_sequences) as numpy arrays.
        X_sequences shape: (num_sequences, sequence_length, num_features)
        y_sequences shape: (num_sequences,) or (num_sequences, num_targets)
    """
    X_sequences = []
    y_sequences = []
    
    # Get feature column names/indices
    if isinstance(feature_cols[0], str):
        feature_names = feature_cols
    else:
        feature_names = df.columns[feature_cols].tolist()
    
    # Get target column names/indices  
    if isinstance(target_cols[0], str):
        target_names = target_cols
    else:
        target_names = df.columns[target_cols].tolist()
    
    # Convert column indices to names if needed
    seq_id_name = df.columns[sequence_id_col] if sequence_id_col is not None else None
    time_col_name = df.columns[time_col] if time_col is not None else None
    
    if seq_id_name is not None:
        # Multiple sequences (e.g., multiple engines)
        for seq_id in df[seq_id_name].unique():
            seq_data = df[df[seq_id_name] == seq_id]
            
            # Sort by time column if specified
            if time_col_name is not None:
                seq_data = seq_data.sort_values(time_col_name)
            
            features = seq_data[feature_names].values
            targets = seq_data[target_names].values
            
            # Create sliding window sequences
            if len(features) >= sequence_length:
                for i in range(len(features) - sequence_length + 1):
                    X_sequences.append(features[i:i + sequence_length])
                    # Target is the value at the end of the sequence
                    y_sequences.append(targets[i + sequence_length - 1])
    else:
        # Single continuous sequence
        if time_col_name is not None:
            df = df.sort_values(time_col_name)
        
        features = df[feature_names].values
        targets = df[target_names].values
        
        # Create sliding window sequences
        for i in range(len(features) - sequence_length + 1):
            X_sequences.append(features[i:i + sequence_length])
            y_sequences.append(targets[i + sequence_length - 1])
    
    return np.array(X_sequences), np.array(y_sequences)

# Load training data
print(f"\nLoading data from '{TRAINING_DATA_PATH}'...")
Training_df = pd.read_csv(TRAINING_DATA_PATH)
print(f"Loaded {len(Training_df)} rows, {len(Training_df.columns)} columns")

# Parse column selections
feature_cols = parse_column_range(FEATURE_COLUMNS, Training_df)
target_cols = parse_column_range(TARGET_COLUMNS, Training_df)

# Get actual column names for display
if isinstance(feature_cols[0], int):
    feature_names = Training_df.columns[feature_cols].tolist()
else:
    feature_names = feature_cols
    
if isinstance(target_cols[0], int):
    target_names = Training_df.columns[target_cols].tolist()
else:
    target_names = target_cols

print(f"Feature columns ({len(feature_names)}): {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
print(f"Target columns: {target_names}")

# Scale feature data if enabled
scaler = None
if SCALE_DATA:
    print("Scaling features to 0-1 range...")
    scaler = MinMaxScaler()
    Training_df[feature_names] = scaler.fit_transform(Training_df[feature_names])

# Split data if using training data as test
if USE_TRAINING_DATA_AS_TEST:
    if SEQUENCE_ID_COLUMN is not None:
        # Split by sequence IDs to avoid data leakage
        seq_id_col_name = Training_df.columns[SEQUENCE_ID_COLUMN]
        unique_ids = Training_df[seq_id_col_name].unique()
        np.random.shuffle(unique_ids)
        split_idx = int(len(unique_ids) * (1 - TRAIN_TEST_SPLIT))
        train_ids = unique_ids[:split_idx]
        test_ids = unique_ids[split_idx:]
        
        train_df = Training_df[Training_df[seq_id_col_name].isin(train_ids)]
        test_df = Training_df[Training_df[seq_id_col_name].isin(test_ids)]
        print(f"Split by column {SEQUENCE_ID_COLUMN} ('{seq_id_col_name}'): {len(train_ids)} train, {len(test_ids)} test")
    else:
        train_df, test_df = train_test_split(Training_df, test_size=TRAIN_TEST_SPLIT, 
                                              random_state=MANUAL_SEED, shuffle=False)
else:
    train_df = Training_df
    test_df = pd.read_csv(TEST_DATA_PATH)
    if SCALE_DATA and scaler is not None:
        test_df[feature_names] = scaler.transform(test_df[feature_names])

# Create sequences
print(f"\nCreating sequences with length {SEQUENCE_LENGTH}...")
X_train, y_train = create_sequences(train_df, feature_cols, target_cols, SEQUENCE_LENGTH,
                                     SEQUENCE_ID_COLUMN, TIME_COLUMN)
X_test, y_test = create_sequences(test_df, feature_cols, target_cols, SEQUENCE_LENGTH,
                                   SEQUENCE_ID_COLUMN, TIME_COLUMN)

# Flatten targets if single column
if y_train.ndim > 1 and y_train.shape[1] == 1:
    y_train = y_train.ravel()
    y_test = y_test.ravel()

print(f"Training sequences: {X_train.shape[0]} sequences, shape {X_train.shape}")
print(f"Testing sequences: {X_test.shape[0]} sequences")

# Convert to PyTorch tensors and move to device
X_train = torch.FloatTensor(X_train).to(DEVICE)
X_test = torch.FloatTensor(X_test).to(DEVICE)

# Convert y targets to appropriate tensor types
if LOSS_FUNCTION in ['cross_entropy', 'nll']:
    y_train = torch.LongTensor(y_train).to(DEVICE)
    y_test = torch.LongTensor(y_test).to(DEVICE)
else:
    y_train = torch.FloatTensor(y_train).to(DEVICE)
    y_test = torch.FloatTensor(y_test).to(DEVICE)

# Create DataLoaders for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize loss criterion and optimizer
loss_criterion = LOSS_FUNCTIONS[LOSS_FUNCTION]()
optimizer = OPTIMIZERS[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

''' ========== MODEL TRAINING ========== '''
print(f"\n{'='*50}")
print("Starting Training...")
print(f"{'='*50}")

losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        y_pred = model(batch_X)
        
        # Compute loss (squeeze for regression with single output)
        if LOSS_FUNCTION not in ['cross_entropy', 'nll'] and y_pred.dim() > 1 and y_pred.size(1) == 1:
            loss = loss_criterion(y_pred.squeeze(), batch_y)
        else:
            loss = loss_criterion(y_pred, batch_y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Average loss for epoch
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss ({RNN_TYPE.upper()})')
plt.grid(True, alpha=0.3)
plt.savefig('rnn_training_loss.png')
print("Training loss plot saved as 'rnn_training_loss.png'")

''' ========== MODEL EVALUATION ========== '''
print(f"\n{'='*50}")
print("Evaluating Model...")
print(f"{'='*50}")

model.eval()
all_predictions = []
all_targets = []
test_loss = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X)
        
        if LOSS_FUNCTION not in ['cross_entropy', 'nll'] and y_pred.dim() > 1 and y_pred.size(1) == 1:
            loss = loss_criterion(y_pred.squeeze(), batch_y)
            all_predictions.append(y_pred.squeeze())
        else:
            loss = loss_criterion(y_pred, batch_y)
            all_predictions.append(y_pred)
        
        test_loss += loss.item()
        all_targets.append(batch_y)

# Concatenate all batches
y_pred_all = torch.cat(all_predictions)
y_test_all = torch.cat(all_targets)
test_loss = test_loss / len(test_loader)

print(f"Test Loss: {test_loss:.4f}")

# Classification metrics
if LOSS_FUNCTION in ['cross_entropy', 'nll']:
    y_pred_classes = torch.argmax(y_pred_all, dim=1)
    correct = (y_pred_classes == y_test_all).sum().item()
    accuracy = correct / y_test_all.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{y_test_all.size(0)} correct)")
    
    # Print first 20 predictions
    print("\nFirst 20 Predictions vs Actual:")
    for i in range(min(20, y_test_all.size(0))):
        print(f"{i+1}.) Predicted: {y_pred_classes[i].item()} \t Actual: {y_test_all[i].item()}")

# Regression metrics
else:
    # Move to CPU for numpy operations
    y_pred_np = y_pred_all.cpu().numpy()
    y_test_np = y_test_all.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((y_pred_np - y_test_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_np - y_test_np))
    
    # R² Score
    ss_res = np.sum((y_test_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n--- Regression Metrics ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Print first 20 predictions
    print("\nFirst 20 Predictions vs Actual:")
    for i in range(min(20, len(y_test_np))):
        if y_pred_np.ndim == 1 or y_pred_np.shape[-1] == 1:
            pred_val = y_pred_np[i] if y_pred_np.ndim == 1 else y_pred_np[i][0]
            actual_val = y_test_np[i] if y_test_np.ndim == 1 else y_test_np[i][0]
            print(f"{i+1}.) Predicted: {pred_val:.4f} \t Actual: {actual_val:.4f}")
        else:
            pred_str = ', '.join([f"{v:.2f}" for v in y_pred_np[i]])
            actual_str = ', '.join([f"{v:.2f}" for v in y_test_np[i]])
            print(f"{i+1}.) Predicted: [{pred_str}] \t Actual: [{actual_str}]")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    sample_size = min(500, len(y_test_np))
    indices = np.random.choice(len(y_test_np), sample_size, replace=False)
    indices = np.sort(indices)
    
    plt.scatter(range(sample_size), y_test_np[indices], label='Actual', alpha=0.7, marker='x')
    plt.scatter(range(sample_size), y_pred_np[indices], label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted ({RNN_TYPE.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rnn_predictions.png')
    print("\nPrediction plot saved as 'rnn_predictions.png'")

# Save the trained model
torch.save(model.state_dict(), SAVE_MODEL_FILE_PATH)
print(f"\nModel saved as '{SAVE_MODEL_FILE_PATH}'")    