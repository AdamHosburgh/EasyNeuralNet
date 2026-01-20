# EasyRNN - Recurrent Neural Network Template

A simple, configurable template for building and training Recurrent Neural Networks (RNNs) using PyTorch. Ideal for time series forecasting, sequence prediction, and any task where the order of data matters.

---

## Table of Contents
- [Quick Start](#quick-start)
- [When to Use RNNs](#when-to-use-rnns)
- [Configuration Parameters](#configuration-parameters)
  - [Basic Settings](#basic-settings)
  - [Architecture Parameters](#architecture-parameters)
  - [Sequence Parameters](#sequence-parameters)
  - [Data Configuration](#data-configuration)
  - [Training Parameters](#training-parameters)
- [Tuning Guide](#tuning-guide)
- [Common Issues](#common-issues)

---

## Quick Start

1. Place your CSV data file in the same directory (or provide full path)
2. Edit the configuration parameters at the top of `EasyRNN.py`
3. Run the script: `python EasyRNN.py`
4. Check the output metrics and saved plots

---

## When to Use RNNs

RNNs are designed for **sequential data** where order matters:

| Use Case | Example |
|----------|---------|
| Time Series Forecasting | Stock prices, weather, sensor readings |
| Predictive Maintenance | Remaining Useful Life (RUL) prediction |
| Sequence Classification | Activity recognition, anomaly detection |
| Natural Language | Sentiment analysis, text classification |

**Don't use RNNs for:** Static tabular data with no temporal relationship (use EasyNN instead).

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

#### `INPUT_SIZE`
```python
INPUT_SIZE = 24
```
**What it does:** The number of features (columns) at each timestep. This must match the number of columns specified in `FEATURE_COLUMNS`.

**How to set it:** Count your feature columns. If `FEATURE_COLUMNS = '2-25'`, that's 24 columns (2,3,4...25), so `INPUT_SIZE = 24`.

---

#### `HIDDEN_SIZE`
```python
HIDDEN_SIZE = 128
```
**What it does:** The number of neurons in each RNN layer. This is the "memory capacity" of your network.

**How to set it:**
| Dataset Size | Complexity | Suggested Hidden Size |
|--------------|------------|----------------------|
| Small (<1K samples) | Simple | 32-64 |
| Medium (1K-10K) | Moderate | 64-128 |
| Large (10K+) | Complex | 128-256 |

⚠️ **Larger isn't always better** — too large can cause overfitting and slower training.

---

#### `NUM_LAYERS`
```python
NUM_LAYERS = 2
```
**What it does:** The number of stacked RNN layers. More layers = deeper network that can learn more complex patterns.

**How to set it:**
- **1 layer:** Good for simple patterns, fastest training
- **2 layers:** Good default, handles most problems well
- **3+ layers:** Only for very complex sequences, requires more data

⚠️ More layers significantly increases training time and risk of overfitting.

---

#### `OUTPUT_SIZE`
```python
OUTPUT_SIZE = 1
```
**What it does:** The number of output values the model predicts.

**How to set it:**
- **Regression (predicting a number):** Usually `1`
- **Multi-output regression:** Number of values to predict
- **Classification:** Number of classes (e.g., 10 for digits 0-9)

---

#### `RNN_TYPE`
```python
RNN_TYPE = 'lstm'
```
**What it does:** Selects the type of recurrent cell architecture.

| Type | Best For | Pros | Cons |
|------|----------|------|------|
| `'lstm'` | Long sequences, complex patterns | Handles long-term dependencies, most stable | Slower, more parameters |
| `'gru'` | Medium sequences, faster training | Faster than LSTM, similar performance | Slightly less capacity |
| `'rnn'` | Very short sequences only | Fastest, simplest | Suffers from vanishing gradients |

**Recommendation:** Start with `'lstm'`. Try `'gru'` if training is too slow.

---

#### `BIDIRECTIONAL`
```python
BIDIRECTIONAL = False
```
**What it does:** When `True`, the RNN processes sequences both forward and backward, then combines the results.

**When to use:**
- `False` (default): For forecasting/prediction where you only have past data
- `True`: For classification tasks where you have the complete sequence (e.g., sentiment of a full sentence)

⚠️ Bidirectional doubles the model size and training time.

---

#### `DROPOUT`
```python
DROPOUT = 0.2
```
**What it does:** Randomly "drops out" (sets to zero) a percentage of neurons during training. This prevents overfitting.

**How to set it:**
| Situation | Suggested Dropout |
|-----------|-------------------|
| Small dataset, overfitting | 0.3 - 0.5 |
| Medium dataset | 0.2 - 0.3 |
| Large dataset | 0.1 - 0.2 |
| No overfitting | 0.0 |

**Signs you need more dropout:** Training loss is much lower than test loss.

---

### Sequence Parameters

#### `SEQUENCE_LENGTH`
```python
SEQUENCE_LENGTH = 50
```
**What it does:** The number of timesteps the model looks back to make a prediction. This is your "lookback window."

**How to set it:**
- Consider how much historical context is needed to make a good prediction
- For RUL prediction: Long enough to capture degradation patterns (50-100 cycles)
- For stock prediction: Days/weeks of historical data

| Too Short | Too Long |
|-----------|----------|
| Misses important patterns | Slower training |
| Faster training | May include irrelevant old data |
| Less memory usage | More memory usage |

**Finding the right value:** Start with a reasonable guess based on your domain knowledge, then experiment.

---

#### `SEQUENCE_ID_COLUMN`
```python
SEQUENCE_ID_COLUMN = 0
```
**What it does:** The column index containing the identifier for each independent sequence (e.g., different engines, patients, sensors).

**How to set it:**
- Set to the column number (0-indexed) containing your sequence IDs
- Set to `None` if your data is one continuous sequence

**Example:** If column 0 contains engine IDs (1, 2, 3...), set `SEQUENCE_ID_COLUMN = 0`

---

#### `TIME_COLUMN`
```python
TIME_COLUMN = 1
```
**What it does:** The column index containing time/order information. The data will be sorted by this column within each sequence.

**How to set it:**
- Set to the column number (0-indexed) containing timestamps or cycle counts
- Set to `None` if your data is already sorted chronologically

---

### Data Configuration

#### `TRAINING_DATA_PATH`
```python
TRAINING_DATA_PATH = 'Train1.csv'
```
**What it does:** Path to your training data CSV file.

**How to set it:** Use a filename (if in same directory) or full path.

---

#### `FEATURE_COLUMNS`
```python
FEATURE_COLUMNS = '2-25'
```
**What it does:** Specifies which columns contain input features.

**Format options:**
- Range: `'2-25'` (columns 2 through 25)
- Single: `'5'` (just column 5)
- Mixed: `'2-10,15,20-25'` (columns 2-10, 15, and 20-25)

⚠️ Don't include the sequence ID, time column, or target in your features!

---

#### `TARGET_COLUMNS`
```python
TARGET_COLUMNS = '26'
```
**What it does:** Specifies which column(s) contain the target values to predict.

**How to set it:** Usually a single column index for regression or classification.

---

#### `USE_TRAINING_DATA_AS_TEST`
```python
USE_TRAINING_DATA_AS_TEST = True
```
**What it does:** When `True`, automatically splits your training data into train/test sets.

**When to use:**
- `True`: You only have one data file
- `False`: You have separate train and test files

---

#### `SCALE_DATA`
```python
SCALE_DATA = True
```
**What it does:** Scales all features to the 0-1 range using Min-Max scaling.

**Recommendation:** Almost always set to `True` for RNNs. Neural networks work better with normalized data.

---

#### `TRAIN_TEST_SPLIT`
```python
TRAIN_TEST_SPLIT = 0.2
```
**What it does:** The fraction of data (or sequences) to use for testing.

**How to set it:** 
- `0.2` = 20% test, 80% train (good default)
- `0.1` = 10% test (if you have lots of data)
- `0.3` = 30% test (if you want more robust evaluation)

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
| `'adamw'` | Preventing overfitting | Adam + weight decay |
| `'sgd'` | Fine-tuning | Simple, may need learning rate scheduling |
| `'rmsprop'` | RNNs specifically | Good alternative to Adam |

**Recommendation:** Start with `'adam'`.

---

#### `LEARNING_RATE`
```python
LEARNING_RATE = 0.001
```
**What it does:** Controls how big of a step the optimizer takes when updating weights.

**How to set it:**
| Learning Rate | Effect |
|---------------|--------|
| Too high (>0.01) | Training unstable, loss may explode |
| Good (0.0001-0.001) | Steady improvement |
| Too low (<0.0001) | Very slow training |

**Recommendation:** Start with `0.001`. If loss is jumpy, try `0.0001`.

---

#### `BATCH_SIZE`
```python
BATCH_SIZE = 32
```
**What it does:** Number of sequences processed together before updating weights.

**How to set it:**
| Batch Size | Pros | Cons |
|------------|------|------|
| Small (8-16) | Better generalization | Slower, noisier gradients |
| Medium (32-64) | Good balance | — |
| Large (128+) | Faster training | May generalize worse, needs more memory |

**Recommendation:** Start with `32`. Increase if you have GPU memory to spare.

---

#### `EPOCHS`
```python
EPOCHS = 100
```
**What it does:** Number of times the model sees the entire training dataset.

**How to set it:**
- Watch the training loss plot
- Stop when loss plateaus (not improving anymore)
- If test loss starts increasing while train loss decreases = overfitting

**Typical values:** 50-200 for most problems.

---

## Tuning Guide

### Step-by-Step Approach

1. **Start simple:** 
   - `NUM_LAYERS = 1`
   - `HIDDEN_SIZE = 64`
   - `DROPOUT = 0.0`

2. **Get a baseline:** Run and note the test metrics.

3. **Increase capacity if underfitting** (train loss stays high):
   - Increase `HIDDEN_SIZE`
   - Add more `NUM_LAYERS`
   - Try longer `SEQUENCE_LENGTH`

4. **Add regularization if overfitting** (train loss << test loss):
   - Increase `DROPOUT`
   - Decrease `HIDDEN_SIZE` or `NUM_LAYERS`
   - Get more training data

5. **Fine-tune:**
   - Adjust `LEARNING_RATE`
   - Try different `RNN_TYPE`
   - Experiment with `SEQUENCE_LENGTH`

---

## Common Issues

### "No training sequences were created"
- `SEQUENCE_LENGTH` is longer than your individual sequences
- `SEQUENCE_ID_COLUMN` is pointing to the wrong column
- Data doesn't have multiple rows per sequence ID

### Training loss not decreasing
- Learning rate too low → increase `LEARNING_RATE`
- Model too simple → increase `HIDDEN_SIZE` or `NUM_LAYERS`
- Data not scaled → set `SCALE_DATA = True`

### Training loss decreases but test loss increases (overfitting)
- Add dropout → increase `DROPOUT`
- Reduce model size → decrease `HIDDEN_SIZE` or `NUM_LAYERS`
- Get more training data

### Out of memory (OOM) error
- Reduce `BATCH_SIZE`
- Reduce `HIDDEN_SIZE`
- Reduce `SEQUENCE_LENGTH`

---

## Output Files

After running, the script produces:
- `rnn_model.pth` — Saved model weights (for loading later)
- `rnn_training_loss.png` — Plot of training loss over epochs
- `rnn_predictions.png` — Plot comparing actual vs predicted values (regression only)

---

## Example Configurations

### Turbofan RUL Prediction
```python
INPUT_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1
RNN_TYPE = 'lstm'
SEQUENCE_LENGTH = 50
SEQUENCE_ID_COLUMN = 0      # engine column
TIME_COLUMN = 1             # cycle column
FEATURE_COLUMNS = '2-25'
TARGET_COLUMNS = '26'       # RUL column
LOSS_FUNCTION = 'mse'
```

### Stock Price Prediction
```python
INPUT_SIZE = 5              # open, high, low, close, volume
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1             # predict next close price
RNN_TYPE = 'lstm'
SEQUENCE_LENGTH = 30        # look back 30 days
SEQUENCE_ID_COLUMN = None   # single continuous sequence
TIME_COLUMN = 0             # date column
FEATURE_COLUMNS = '1-5'
TARGET_COLUMNS = '4'        # close price
LOSS_FUNCTION = 'mse'
```

### Sequence Classification (e.g., Activity Recognition)
```python
INPUT_SIZE = 6              # accelerometer x,y,z + gyroscope x,y,z
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 6             # 6 activity classes
RNN_TYPE = 'lstm'
BIDIRECTIONAL = True        # have full sequence for classification
SEQUENCE_LENGTH = 100
SEQUENCE_ID_COLUMN = 0      # participant/trial ID
TIME_COLUMN = 1
FEATURE_COLUMNS = '2-7'
TARGET_COLUMNS = '8'        # activity label
LOSS_FUNCTION = 'cross_entropy'
```