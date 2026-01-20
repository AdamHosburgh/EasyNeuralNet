# EasyNN - Multilayer Perceptron (MLP) Neural Network Template

A simple, configurable template for building and training feedforward neural networks (MLPs) using PyTorch. Ideal for classification and regression tasks on tabular/structured data.

---

## Table of Contents
- [Quick Start](#quick-start)
- [When to Use MLPs](#when-to-use-mlps)
- [Configuration Parameters](#configuration-parameters)
  - [Basic Settings](#basic-settings)
  - [Architecture Parameters](#architecture-parameters)
  - [Data Configuration](#data-configuration)
  - [Training Parameters](#training-parameters)
- [Tuning Guide](#tuning-guide)
- [Common Issues](#common-issues)

---

## Quick Start

1. Place your CSV data file in the same directory (or provide full path)
2. Edit the configuration parameters at the top of `EasyNN.py`
3. Run the script: `python EasyNN.py`
4. Check the output metrics and saved plots

---

## When to Use MLPs

MLPs (Multilayer Perceptrons) are designed for **tabular/structured data** where each sample is independent:

| Use Case | Example |
|----------|---------|
| Classification | Spam detection, disease diagnosis, image classification (flattened) |
| Regression | Price prediction, score estimation |
| Tabular Data | CSV files with features as columns, samples as rows |
| Feature-based Learning | When you have extracted features (not raw sequences) |

**Don't use MLPs for:** Sequential data where order matters (use EasyRNN instead), or image data where spatial relationships matter (use CNNs).

---

## Configuration Parameters

### Basic Settings

#### `MANUAL_SEED`
```python
MANUAL_SEED = 42
```
**What it does:** Sets the random seed for reproducibility. Using the same seed ensures you get identical results each run.

**How to set it:** Any integer works. Change it only if you want different random initialization.

---

### Architecture Parameters

#### `IN_FEATURES`
```python
IN_FEATURES = 63
```
**What it does:** The number of input features (columns) your model expects. This must match the number of columns specified in `FEATURE_COLUMNS`.

**How to set it:** Count your feature columns. If `FEATURE_COLUMNS = '0-62'`, that's 63 columns (0,1,2...62), so `IN_FEATURES = 63`.

---

#### `HIDDEN_LAYERS`
```python
HIDDEN_LAYERS = [128, 64, 32, 16]
```
**What it does:** Defines the architecture of your neural network. Each number represents the number of neurons in that hidden layer.

**How to set it:**
| Pattern | Example | Use Case |
|---------|---------|----------|
| Decreasing (funnel) | `[128, 64, 32]` | Most common, compresses information toward output |
| Constant | `[64, 64, 64]` | Maintains information capacity throughout |
| Increasing | `[32, 64, 128]` | Expands representation (less common) |
| Bottleneck | `[128, 32, 128]` | Forces compression then expansion (autoencoder-style) |

**Guidelines:**
| Dataset Size | Suggested Architecture |
|--------------|----------------------|
| Small (<1K samples) | `[32, 16]` or `[64, 32]` |
| Medium (1K-10K) | `[128, 64, 32]` |
| Large (10K-100K) | `[256, 128, 64, 32]` |
| Very Large (100K+) | `[512, 256, 128, 64]` |

⚠️ **More layers ≠ better.** Start simple and add complexity only if needed.

---

#### `OUT_FEATURES`
```python
OUT_FEATURES = 8
```
**What it does:** The number of output values the model produces.

**How to set it:**
| Task | OUT_FEATURES |
|------|--------------|
| Binary classification | `1` (with `bce_logits`) or `2` (with `cross_entropy`) |
| Multi-class classification | Number of classes (e.g., `10` for digits 0-9) |
| Regression (single value) | `1` |
| Multi-output regression | Number of values to predict |

---

#### `DROPOUT`
```python
DROPOUT = 0.2
```
**What it does:** Randomly "drops out" (sets to zero) a percentage of neurons during training. This prevents overfitting by forcing the network to not rely on any single neuron.

**How to set it:**
| Situation | Suggested Dropout |
|-----------|-------------------|
| Small dataset, overfitting | 0.3 - 0.5 |
| Medium dataset | 0.2 - 0.3 |
| Large dataset | 0.1 - 0.2 |
| No overfitting observed | 0.0 |

**Signs you need more dropout:** Training accuracy is much higher than test accuracy.

---

#### `ACTIVATION`
```python
ACTIVATION = 'relu'
```
**What it does:** The non-linear function applied after each hidden layer. This allows the network to learn complex patterns.

| Activation | Best For | Pros | Cons |
|------------|----------|------|------|
| `'relu'` | General use (default) | Fast, simple, works well | Can "die" (output 0 forever) |
| `'leaky_relu'` | When ReLU neurons die | Fixes dying ReLU problem | Slightly more computation |
| `'gelu'` | Transformer-style models | Smooth, modern | Slower than ReLU |
| `'elu'` | Deeper networks | Smooth, handles negatives well | Slower than ReLU |
| `'silu'` | Modern architectures | Self-gated, smooth | Newer, less tested |
| `'mish'` | Experimental | Self-regularizing | Newest, computationally heavier |

**Recommendation:** Start with `'relu'`. Try `'leaky_relu'` if training stalls.

---

#### `BATCH_NORM`
```python
BATCH_NORM = False
```
**What it does:** Normalizes the outputs of each layer to have zero mean and unit variance. This can stabilize and speed up training.

**When to use:**
| Situation | BATCH_NORM |
|-----------|------------|
| Deep networks (4+ layers) | `True` |
| Training is unstable | `True` |
| Small batch sizes (<16) | `False` (batch stats unreliable) |
| Simple/shallow networks | `False` |

**Trade-off:** Adds slight computational overhead but can allow higher learning rates and faster convergence.

---

### Loss Functions

#### `LOSS_FUNCTION`
```python
LOSS_FUNCTION = 'cross_entropy'
```
**What it does:** Defines how the model measures its prediction errors.

**Classification losses:**
| Loss | Use Case | Notes |
|------|----------|-------|
| `'cross_entropy'` | Multi-class classification | Most common, outputs are class probabilities |
| `'bce_logits'` | Binary classification | Numerically stable, use with 1 output |
| `'bce'` | Binary classification | Requires sigmoid output |
| `'nll'` | Multi-class | Requires log_softmax output |

**Regression losses:**
| Loss | Use Case | Notes |
|------|----------|-------|
| `'mse'` | General regression | Penalizes large errors heavily |
| `'l1'` | Robust regression | Less sensitive to outliers than MSE |
| `'smooth_l1'` | Balanced regression | Combines MSE and L1 benefits |
| `'huber'` | Outlier-resistant | Similar to smooth_l1 |

**Quick guide:**
- Classification → `'cross_entropy'`
- Binary classification → `'bce_logits'`
- Regression → `'mse'`
- Regression with outliers → `'huber'` or `'smooth_l1'`

---

### Data Configuration

#### `TRAINING_DATA_PATH`
```python
TRAINING_DATA_PATH = 'Train.csv'
```
**What it does:** Path to your training data CSV file.

**How to set it:** Use a filename (if in same directory) or full path like `'/path/to/data.csv'`.

---

#### `FEATURE_COLUMNS`
```python
FEATURE_COLUMNS = '0-62'
```
**What it does:** Specifies which columns contain input features.

**Format options:**
- Range: `'0-62'` (columns 0 through 62)
- Single: `'5'` (just column 5)
- Mixed: `'0-10,15,20-25'` (columns 0-10, 15, and 20-25)

⚠️ Don't include the target column in your features!

---

#### `TARGET_COLUMNS`
```python
TARGET_COLUMNS = '63'
```
**What it does:** Specifies which column(s) contain the target values to predict.

**How to set it:** Usually a single column index for classification or regression.

---

#### `USE_TRAINING_DATA_AS_TEST`
```python
USE_TRAINING_DATA_AS_TEST = True
```
**What it does:** When `True`, automatically splits your training data into train/test sets.

**When to use:**
- `True`: You only have one data file
- `False`: You have separate train and test files (set `TEST_DATA_PATH`)

---

#### `SCALE_DATA`
```python
SCALE_DATA = False
```
**What it does:** Scales all features to the 0-1 range using Min-Max scaling.

**When to use:**
| Situation | SCALE_DATA |
|-----------|------------|
| Features have very different scales | `True` |
| Features are already normalized | `False` |
| Using features like pixel values (0-255) | `True` |
| Features are all similar scale | `False` (but `True` doesn't hurt) |

**Recommendation:** When in doubt, set to `True`. Scaling rarely hurts and often helps.

---

#### `TRAIN_TEST_SPLIT`
```python
TRAIN_TEST_SPLIT = 0.15
```
**What it does:** The fraction of data to use for testing (when `USE_TRAINING_DATA_AS_TEST = True`).

**How to set it:**
- `0.15` = 15% test, 85% train
- `0.2` = 20% test, 80% train (common default)
- `0.1` = 10% test (if you have lots of data)

---

### Training Parameters

#### `OPTIMIZER`
```python
OPTIMIZER = 'adam'
```
**What it does:** The algorithm used to update weights during training.

| Optimizer | Best For | Notes |
|-----------|----------|-------|
| `'adam'` | General use | Great default, adaptive learning rate |
| `'adamw'` | Preventing overfitting | Adam + weight decay regularization |
| `'sgd'` | Fine-tuning, large datasets | Simple, may need LR scheduling |
| `'rmsprop'` | Non-stationary problems | Good for RNNs |
| `'nadam'` | Faster convergence | Adam + Nesterov momentum |
| `'radam'` | Stable early training | Rectified Adam |

**Recommendation:** Start with `'adam'`. Try `'adamw'` if overfitting.

---

#### `LEARNING_RATE`
```python
LEARNING_RATE = 0.01
```
**What it does:** Controls how big of a step the optimizer takes when updating weights.

**How to set it:**
| Learning Rate | Effect |
|---------------|--------|
| Too high (>0.1) | Training unstable, loss explodes or oscillates |
| High (0.01-0.1) | Fast learning, may overshoot |
| Medium (0.001-0.01) | Good balance for most cases |
| Low (0.0001-0.001) | Slow but steady, fine-tuning |
| Too low (<0.0001) | Extremely slow training |

**Typical values by optimizer:**
- Adam/AdamW: `0.001` to `0.01`
- SGD: `0.01` to `0.1`

**Tip:** If loss is very jumpy, reduce learning rate. If loss decreases too slowly, increase it.

---

#### `EPOCHS`
```python
EPOCHS = 1000
```
**What it does:** Number of times the model sees the entire training dataset.

**How to set it:**
- Watch the training loss plot
- Stop when loss plateaus (not improving)
- If test loss increases while train loss decreases → overfitting (reduce epochs)

**Typical values:**
| Dataset Size | Suggested Epochs |
|--------------|-----------------|
| Small (<1K) | 500-2000 |
| Medium (1K-10K) | 200-1000 |
| Large (10K+) | 50-200 |

---

#### `SAVE_MODEL_FILE_PATH`
```python
SAVE_MODEL_FILE_PATH = 'model.pth'
```
**What it does:** Where to save the trained model weights for later use.

---

## Tuning Guide

### Step-by-Step Approach

1. **Start simple:**
   ```python
   HIDDEN_LAYERS = [64, 32]
   DROPOUT = 0.0
   BATCH_NORM = False
   ACTIVATION = 'relu'
   LEARNING_RATE = 0.001
   ```

2. **Get a baseline:** Run and note the test metrics.

3. **If underfitting** (both train and test loss are high):
   - Add more neurons: `[128, 64, 32]`
   - Add more layers: `[128, 64, 32, 16]`
   - Increase learning rate
   - Train for more epochs

4. **If overfitting** (train loss << test loss):
   - Add dropout: `DROPOUT = 0.3`
   - Remove layers or neurons
   - Enable batch normalization
   - Get more training data
   - Reduce epochs

5. **Fine-tune:**
   - Try different activations
   - Adjust learning rate
   - Experiment with optimizers

### Architecture Design Tips

**For Classification:**
```python
# Example: 100 features → 10 classes
IN_FEATURES = 100
HIDDEN_LAYERS = [64, 32]    # Funnel down
OUT_FEATURES = 10
LOSS_FUNCTION = 'cross_entropy'
```

**For Regression:**
```python
# Example: 50 features → 1 value
IN_FEATURES = 50
HIDDEN_LAYERS = [64, 32, 16]
OUT_FEATURES = 1
LOSS_FUNCTION = 'mse'
```

**For Complex Problems:**
```python
# More capacity with regularization
HIDDEN_LAYERS = [256, 128, 64, 32]
DROPOUT = 0.3
BATCH_NORM = True
```

---

## Common Issues

### Training loss not decreasing
- Learning rate too low → increase `LEARNING_RATE`
- Model too simple → add more layers/neurons to `HIDDEN_LAYERS`
- Data not scaled → set `SCALE_DATA = True`
- Wrong loss function → check `LOSS_FUNCTION` matches your task

### Training loss decreasing but test loss increases (overfitting)
- Add regularization → increase `DROPOUT` to 0.3-0.5
- Reduce model complexity → fewer/smaller layers in `HIDDEN_LAYERS`
- Enable batch normalization → `BATCH_NORM = True`
- Reduce epochs → stop training earlier
- Get more training data

### Loss is NaN or explodes
- Learning rate too high → reduce `LEARNING_RATE`
- Data contains NaN values → clean your data
- Try gradient clipping or batch normalization

### Poor accuracy despite low loss
- Check if classes are imbalanced
- Verify `OUT_FEATURES` matches number of classes
- Make sure `LOSS_FUNCTION` is appropriate for your task

### "CUDA out of memory" error
- Reduce `HIDDEN_LAYERS` sizes
- Process data in smaller batches (modify code to use DataLoader)

---

## Output Files

After running, the script produces:
- `model.pth` — Saved model weights (for loading later)
- `training_loss.png` — Plot of training loss over epochs

---

## Example Configurations

### Binary Classification (e.g., Spam Detection)
```python
IN_FEATURES = 100           # 100 text features
HIDDEN_LAYERS = [64, 32]
OUT_FEATURES = 2            # spam or not spam
DROPOUT = 0.3
ACTIVATION = 'relu'
LOSS_FUNCTION = 'cross_entropy'
LEARNING_RATE = 0.001
EPOCHS = 500
```

### Multi-class Classification (e.g., Digit Recognition)
```python
IN_FEATURES = 784           # 28x28 flattened image
HIDDEN_LAYERS = [256, 128, 64]
OUT_FEATURES = 10           # digits 0-9
DROPOUT = 0.2
ACTIVATION = 'relu'
BATCH_NORM = True
LOSS_FUNCTION = 'cross_entropy'
LEARNING_RATE = 0.001
EPOCHS = 100
```

### Regression (e.g., House Price Prediction)
```python
IN_FEATURES = 13            # 13 housing features
HIDDEN_LAYERS = [64, 32, 16]
OUT_FEATURES = 1            # price
DROPOUT = 0.1
ACTIVATION = 'relu'
LOSS_FUNCTION = 'mse'
SCALE_DATA = True           # important for regression
LEARNING_RATE = 0.001
EPOCHS = 1000
```

### Large Dataset with Complex Patterns
```python
IN_FEATURES = 500
HIDDEN_LAYERS = [512, 256, 128, 64, 32]
OUT_FEATURES = 20
DROPOUT = 0.4
ACTIVATION = 'leaky_relu'
BATCH_NORM = True
LOSS_FUNCTION = 'cross_entropy'
OPTIMIZER = 'adamw'
LEARNING_RATE = 0.0005
EPOCHS = 200
```

---

## Loading a Saved Model

To use your trained model later:

```python
import torch

# Recreate the model architecture (must match training config)
model = Model(in_features=63, hl=[128, 64, 32, 16], out_features=8)

# Load the saved weights
model.load_state_dict(torch.load('model.pth'))

# Set to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    prediction = model(your_input_tensor)
```
