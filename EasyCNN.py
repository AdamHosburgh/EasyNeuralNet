"""
EasyCNN - Convolutional Neural Network Template

A simple, configurable template for building and training Convolutional Neural
Networks using PyTorch. Ideal for image classification and pattern recognition
tasks where spatial relationships matter.

Use cases:
    - Image classification (cats vs dogs, digit recognition, medical imaging)
    - Pattern recognition (texture classification, defect detection)
    - Feature extraction from visual data

Code assistance provided by Claude (Anthropic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

''' ========== USER CONFIGURATION (Edit these as needed) ========== '''

# Random seed for reproducibility (any integer works)
MANUAL_SEED = 42

# ===== Input Image Configuration =====
INPUT_CHANNELS = 1      # 1 for grayscale, 3 for RGB
INPUT_HEIGHT = 28      # Image height in pixels
INPUT_WIDTH = 28       # Image width in pixels

# ===== CNN Architecture Parameters =====
# Each number represents output channels for that conv layer
# Example: [32, 64, 128] creates 3 conv layers with 32, 64, 128 filters
CONV_LAYERS = [32, 64, 128]

# Convolution settings
KERNEL_SIZE = 3         # Size of convolutional kernel (3x3 is standard)
CONV_STRIDE = 1         # Stride for convolution (1 is standard)
CONV_PADDING = 1        # Padding for convolution (1 with kernel=3 preserves size)

# Pooling settings
POOL_TYPE = 'max'       # 'max' or 'avg'
POOL_SIZE = 2           # Size of pooling window (2x2 is standard)
POOL_STRIDE = 2         # Stride for pooling (2 halves dimensions)

# Fully connected layers after convolution (same as EasyNN's HIDDEN_LAYERS)
FC_LAYERS = [512, 256]

# Number of output classes
NUM_CLASSES = 10

# ===== Regularization =====
# Dropout rate for regularization (0 to disable, 0.1-0.5 typical)
DROPOUT = 0.5

# Activation function ('relu', 'leaky_relu', 'elu', 'silu', 'gelu', 'mish')
ACTIVATION = 'relu'

# Batch Normalization flag (recommended for CNNs)
BATCH_NORM = True

# ===== Loss Function =====
# Classification: 'cross_entropy', 'bce_logits', 'nll'
LOSS_FUNCTION = 'cross_entropy'

# ===== Data Configuration =====
# Path to training data folder (ImageFolder format)
# Structure: train_folder/class_name/image.jpg
TRAIN_DATA_PATH = 'data/train'

# Path to test/validation data folder
TEST_DATA_PATH = 'data/test'

# Data augmentation for training (helps prevent overfitting)
USE_DATA_AUGMENTATION = True

# Normalize images using ImageNet statistics (recommended for pretrained-compatible models)
NORMALIZE_IMAGENET = False

# ===== Training Parameters =====
OPTIMIZER = 'adam'      # 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'nadam', 'radam'
LEARNING_RATE = 0.001   # Step size for optimizer (0.001 is good default)
BATCH_SIZE = 32         # Number of images per training batch
EPOCHS = 50             # Number of training epochs

# Path to save the trained model
SAVE_MODEL_FILE_PATH = 'cnn_model.pth'

''' ========== MODEL DEFINITION (Do not edit below) ========== '''

# Device configuration (auto-detects GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    """
    Flexible CNN model with configurable convolutional and fully connected layers.

    Architecture: [Conv -> BatchNorm -> Activation -> Pool] x N -> Flatten -> FC layers

    Input shape: (batch_size, channels, height, width)
    Output shape: (batch_size, num_classes)
    """

    ACTIVATIONS = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'mish': nn.Mish,
    }

    POOL_TYPES = {
        'max': nn.MaxPool2d,
        'avg': nn.AvgPool2d,
    }

    def __init__(self, input_channels=INPUT_CHANNELS, input_height=INPUT_HEIGHT,
                 input_width=INPUT_WIDTH, conv_layers=CONV_LAYERS, fc_layers=FC_LAYERS,
                 num_classes=NUM_CLASSES, kernel_size=KERNEL_SIZE, conv_stride=CONV_STRIDE,
                 conv_padding=CONV_PADDING, pool_type=POOL_TYPE, pool_size=POOL_SIZE,
                 pool_stride=POOL_STRIDE, dropout=DROPOUT, activation=ACTIVATION,
                 batch_norm=BATCH_NORM):
        """
        Initialize the CNN model.

        Args:
            input_channels: Number of input channels (1=grayscale, 3=RGB).
            input_height: Height of input images.
            input_width: Width of input images.
            conv_layers: List of output channels for each conv layer.
            fc_layers: List of neurons for each fully connected layer.
            num_classes: Number of output classes.
            kernel_size: Size of convolutional kernels.
            conv_stride: Stride for convolution operations.
            conv_padding: Padding for convolution operations.
            pool_type: Type of pooling ('max' or 'avg').
            pool_size: Size of pooling window.
            pool_stride: Stride for pooling operations.
            dropout: Dropout rate for regularization.
            activation: Activation function name.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        # Validate parameters
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Activation must be one of {list(self.ACTIVATIONS.keys())}")
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f"Pool type must be one of {list(self.POOL_TYPES.keys())}")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout must be >= 0 and < 1")

        # Store configuration
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels

        for out_channels in conv_layers:
            block = nn.ModuleList()

            # Convolutional layer
            block.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                padding=conv_padding
            ))

            # Batch normalization (if enabled)
            if batch_norm:
                block.append(nn.BatchNorm2d(out_channels))

            # Activation function
            block.append(self.ACTIVATIONS[activation]())

            # Pooling layer
            block.append(self.POOL_TYPES[pool_type](pool_size, pool_stride))

            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate flattened size by passing dummy tensor through conv layers
        self._flat_size = self._get_flat_size(input_channels, input_height, input_width)

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn = nn.ModuleList() if batch_norm else None

        fc_input = self._flat_size
        for fc_output in fc_layers:
            self.fc_layers.append(nn.Linear(fc_input, fc_output))
            if batch_norm:
                self.fc_bn.append(nn.BatchNorm1d(fc_output))
            fc_input = fc_output

        # Output layer
        self.fc_out = nn.Linear(fc_input, num_classes)

        # Activation and dropout for FC layers
        self.activation = self.ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _get_flat_size(self, channels, height, width):
        """Calculate the flattened size after all conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            for block in self.conv_blocks:
                for layer in block:
                    dummy = layer(dummy)
            return dummy.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Pass through convolutional blocks
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.batch_norm and self.fc_bn is not None:
                x = self.fc_bn[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer (no activation - raw logits for cross_entropy)
        x = self.fc_out(x)
        return x


# Set random seed for reproducibility
torch.manual_seed(MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(MANUAL_SEED)

# Print device info
print(f"Using device: {DEVICE}")

# Initialize model and move to device
model = CNNModel().to(DEVICE)

# Print model summary
print(f"\nModel Architecture:")
print(f"  Input: {INPUT_CHANNELS}x{INPUT_HEIGHT}x{INPUT_WIDTH}")
print(f"  Conv layers: {CONV_LAYERS}")
print(f"  FC layers: {FC_LAYERS}")
print(f"  Output classes: {NUM_CLASSES}")
print(f"  Batch norm: {BATCH_NORM}")
print(f"  Dropout: {DROPOUT}")
print(f"  Activation: {ACTIVATION}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

''' ========== DATA LOADING AND PREPROCESSING ========== '''

# Available loss functions
LOSS_FUNCTIONS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce_logits': nn.BCEWithLogitsLoss,
    'nll': nn.NLLLoss,
}

# Validate loss function choice
if LOSS_FUNCTION not in LOSS_FUNCTIONS:
    raise ValueError(f"Loss function must be one of {list(LOSS_FUNCTIONS.keys())}")

# Available optimizers
OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
    'adagrad': torch.optim.Adagrad,
    'nadam': torch.optim.NAdam,
    'radam': torch.optim.RAdam,
}

# Validate optimizer choice
if OPTIMIZER not in OPTIMIZERS:
    raise ValueError(f"Optimizer must be one of {list(OPTIMIZERS.keys())}")

# Build transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
) if NORMALIZE_IMAGENET and INPUT_CHANNELS == 3 else transforms.Normalize(
    mean=[0.5] * INPUT_CHANNELS,
    std=[0.5] * INPUT_CHANNELS
)

# Build base transforms list
base_transforms = [transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH))]

# Add grayscale conversion if needed
if INPUT_CHANNELS == 1:
    base_transforms.append(transforms.Grayscale(num_output_channels=1))

# Training transforms (with optional augmentation)
if USE_DATA_AUGMENTATION:
    aug_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ]
    # ColorJitter only for RGB images
    if INPUT_CHANNELS == 3:
        aug_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    else:
        aug_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    
    train_transform = transforms.Compose(base_transforms + aug_transforms + [transforms.ToTensor(), normalize])
else:
    train_transform = transforms.Compose(base_transforms + [transforms.ToTensor(), normalize])

# Test transforms (no augmentation)
test_transform = transforms.Compose(base_transforms + [transforms.ToTensor(), normalize])

# Load datasets
print(f"\nLoading data from '{TRAIN_DATA_PATH}'...")
try:
    train_dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=test_transform)

    # Get class names
    class_names = train_dataset.classes
    print(f"Found {len(train_dataset)} training images")
    print(f"Found {len(test_dataset)} test images")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

except FileNotFoundError as e:
    print(f"\nError: Data folder not found!")
    print(f"Expected folder structure:")
    print(f"  {TRAIN_DATA_PATH}/")
    print(f"      class1/")
    print(f"          image1.jpg")
    print(f"          image2.jpg")
    print(f"      class2/")
    print(f"          image3.jpg")
    print(f"          ...")
    print(f"\nPlease create the data folders and add images, then run again.")
    raise SystemExit(1)

# Verify NUM_CLASSES matches dataset
if len(class_names) != NUM_CLASSES:
    print(f"\nWarning: NUM_CLASSES ({NUM_CLASSES}) doesn't match dataset ({len(class_names)} classes)")
    print(f"Updating NUM_CLASSES to {len(class_names)}")
    NUM_CLASSES = len(class_names)
    model = CNNModel(num_classes=NUM_CLASSES).to(DEVICE)

# Initialize loss criterion and optimizer
loss_criterion = LOSS_FUNCTIONS[LOSS_FUNCTION]()
optimizer = OPTIMIZERS[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

''' ========== MODEL TRAINING ========== '''

import time

print(f"\n{'=' * 50}")
print("Starting Training...")
print(f"{'=' * 50}")

# Show training overview
num_batches = len(train_loader)
print(f"  Training samples: {len(train_dataset):,}")
print(f"  Batches per epoch: {num_batches}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total epochs: {EPOCHS}")
print(f"{'=' * 50}\n")

train_losses = []
train_accuracies = []
training_start_time = time.time()

# Calculate how often to print batch progress (every ~25% of batches, min 1)
print_every = max(1, num_batches // 4)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    epoch_start_time = time.time()

    print(f"Epoch {epoch + 1}/{EPOCHS} started...")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = loss_criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print batch progress
        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
            progress = 100 * (batch_idx + 1) / num_batches
            current_loss = epoch_loss / (batch_idx + 1)
            current_acc = 100 * correct / total
            print(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.0f}%) - "
                  f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")

    # Calculate epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    # Estimate remaining time
    elapsed_total = time.time() - training_start_time
    avg_epoch_time = elapsed_total / (epoch + 1)
    remaining_epochs = EPOCHS - (epoch + 1)
    eta_seconds = avg_epoch_time * remaining_epochs

    # Format time nicely
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    print(f"  ✓ Epoch {epoch + 1} complete in {format_time(epoch_time)} | "
          f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
          f"ETA: {format_time(eta_seconds)}\n")

total_time = time.time() - training_start_time
print(f"Training complete! Total time: {format_time(total_time)}")

# Plot training metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(train_accuracies)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_metrics.png')
print("Training metrics plot saved as 'cnn_training_metrics.png'")

''' ========== MODEL EVALUATION ========== '''

print(f"\n{'=' * 50}")
print("Evaluating Model...")
print(f"{'=' * 50}")

model.eval()
test_loss = 0
correct = 0
total = 0
all_predictions = []
all_labels = []

# Per-class accuracy tracking
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = loss_criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store for confusion analysis
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Per-class accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1

# Calculate overall metrics
avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total} correct)")

# Print per-class accuracy
print("\n--- Per-Class Accuracy ---")
for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

# Save the trained model
torch.save(model.state_dict(), SAVE_MODEL_FILE_PATH)
print(f"\nModel saved as '{SAVE_MODEL_FILE_PATH}'")

# Save class names for inference
class_info = {
    'class_names': class_names,
    'input_channels': INPUT_CHANNELS,
    'input_height': INPUT_HEIGHT,
    'input_width': INPUT_WIDTH,
}
torch.save(class_info, 'cnn_class_info.pth')
print("Class info saved as 'cnn_class_info.pth'")
