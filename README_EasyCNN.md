# EasyCNN - Convolutional Neural Network Template

A simple, configurable template for building and training Convolutional Neural Networks (CNNs) using PyTorch. Ideal for image classification and pattern recognition tasks where spatial relationships matter.

---

## Table of Contents
- [Quick Start](#quick-start)
- [When to Use CNNs](#when-to-use-cnns)
- [Data Format](#data-format)
- [Configuration Parameters](#configuration-parameters)
  - [Basic Settings](#basic-settings)
  - [Input Configuration](#input-configuration)
  - [Architecture Parameters](#architecture-parameters)
  - [Data Configuration](#data-configuration)
  - [Training Parameters](#training-parameters)
- [Tuning Guide](#tuning-guide)
- [Common Issues](#common-issues)

---

## Quick Start

1. Organize your images into folders by class (see [Data Format](#data-format))
2. Edit the configuration parameters at the top of `EasyCNN.py`
3. Run the script: `python EasyCNN.py`
4. Check the output metrics and saved plots

---

## When to Use CNNs

CNNs are designed for **spatial data** where local patterns and their arrangements matter:

| Use Case | Example |
|----------|---------|
| Image Classification | Cats vs dogs, digit recognition, medical imaging |
| Defect Detection | Manufacturing quality control, crack detection |
| Medical Imaging | X-ray classification, tumor detection, skin lesion analysis |
| Pattern Recognition | Texture classification, satellite imagery |

**Don't use CNNs for:** Tabular data (use EasyNN instead), or sequential data where time order matters (use EasyRNN instead).

---

## Data Format

EasyCNN expects images organized in **ImageFolder format** — folders named by class containing the images:

```
data/
├── train/
│   ├── cat/
│   │   ├── cat001.jpg
│   │   ├── cat002.jpg
│   │   └── ...
│   ├── dog/
│   │   ├── dog001.jpg
│   │   ├── dog002.jpg
│   │   └── ...
│   └── bird/
│       └── ...
└── test/
    ├── cat/
    │   └── ...
    ├── dog/
    │   └── ...
    └── bird/
        └── ...
```

**Key points:**
- Folder names become class labels automatically
- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`
- Images are automatically resized to `INPUT_HEIGHT` × `INPUT_WIDTH`
- Train and test folders should have the same class subfolders

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

### Input Configuration

#### `INPUT_CHANNELS`
```python
INPUT_CHANNELS = 3
```
**What it does:** The number of color channels in your input images.

**How to set it:**
| Value | Image Type |
|-------|------------|
| `1` | Grayscale (black & white) |
| `3` | RGB (color) |

**Note:** Color images are automatically converted. If you set `INPUT_CHANNELS = 1`, RGB images will be converted to grayscale.

---

#### `INPUT_HEIGHT` and `INPUT_WIDTH`
```python
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
```
**What it does:** The size (in pixels) that all images will be resized to before processing.

**How to set it:**
| Size | Use Case | Notes |
|------|----------|-------|
| `32×32` | Small images (CIFAR-style) | Fast training, limited detail |
| `64×64` | Simple patterns | Good balance for small datasets |
| `128×128` | Medium complexity | Common for medical imaging |
| `224×224` | Standard (ImageNet-style) | Good default for most tasks |
| `299×299` | High detail needed | Slower training |

**Guidelines:**
- Larger images = more detail but slower training and more memory
- Square images (height = width) are most common
- `224×224` is a good default that works for most problems

⚠️ **Memory warning:** Doubling image size quadruples memory usage!

---

### Architecture Parameters

#### `CONV_LAYERS`
```python
CONV_LAYERS = [32, 64, 128]
```
**What it does:** Defines the convolutional layers. Each number is the number of filters (output channels) for that layer.

**How it works:**
- First layer: 32 filters detect basic edges, colors, textures
- Second layer: 64 filters combine basics into shapes, patterns
- Third layer: 128 filters detect complex features, object parts

**How to set it:**
| Dataset Complexity | Suggested Architecture |
|-------------------|----------------------|
| Simple (few classes, clear differences) | `[16, 32]` |
| Medium | `[32, 64, 128]` |
| Complex (many classes, subtle differences) | `[64, 128, 256, 512]` |

**Pattern:** Usually increase channels as you go deeper (each layer sees more abstract features).

⚠️ More layers and filters = more parameters = more data needed to train well.

---

#### `KERNEL_SIZE`
```python
KERNEL_SIZE = 3
```
**What it does:** The size of the sliding window (filter) that scans across the image. A `3` means a 3×3 pixel window.

**How to set it:**
| Size | Detects | Use Case |
|------|---------|----------|
| `3` | Fine details, edges | Most common, good default |
| `5` | Larger patterns | When 3×3 misses important features |
| `7` | Very large patterns | First layer of very deep networks |

**Recommendation:** Use `3` for most cases. It's computationally efficient and can capture complex patterns when stacked.

---

#### `CONV_STRIDE`
```python
CONV_STRIDE = 1
```
**What it does:** How many pixels the kernel moves between each operation.

**How to set it:**
| Stride | Effect |
|--------|--------|
| `1` | Kernel moves 1 pixel at a time, preserves spatial detail (standard) |
| `2` | Kernel jumps 2 pixels, reduces output size by half (sometimes used instead of pooling) |

**Recommendation:** Keep at `1`. Use pooling to reduce dimensions instead.

---

#### `CONV_PADDING`
```python
CONV_PADDING = 1
```
**What it does:** Adds zeros around the image border so the output size matches the input size (when stride=1).

**How to set it:**
| Kernel Size | Padding to Preserve Size |
|-------------|-------------------------|
| 3 | 1 |
| 5 | 2 |
| 7 | 3 |

**Formula:** `padding = (kernel_size - 1) / 2`

**Recommendation:** Use `1` with `KERNEL_SIZE = 3` to preserve spatial dimensions through conv layers.

---

#### `POOL_TYPE`
```python
POOL_TYPE = 'max'
```
**What it does:** The type of pooling operation applied after each conv layer to reduce spatial dimensions.

| Type | How It Works | Best For |
|------|--------------|----------|
| `'max'` | Takes maximum value in each window | Most common, preserves strongest activations |
| `'avg'` | Takes average value in each window | Smoother, sometimes better for regression |

**Recommendation:** Use `'max'` for classification tasks.

---

#### `POOL_SIZE` and `POOL_STRIDE`
```python
POOL_SIZE = 2
POOL_STRIDE = 2
```
**What it does:** `POOL_SIZE` is the window size, `POOL_STRIDE` is how far it moves. With both at `2`, each pooling layer halves the image dimensions.

**Example:** 224×224 → 112×112 → 56×56 → 28×28 (after 3 pooling layers)

**How to set it:**
| Setting | Effect |
|---------|--------|
| `2, 2` | Halves dimensions each layer (standard) |
| `3, 2` | Overlapping pooling, smoother reduction |
| `2, 1` | Minimal reduction (rarely used) |

**Recommendation:** Keep both at `2` for standard behavior.

---

#### `FC_LAYERS`
```python
FC_LAYERS = [512, 256]
```
**What it does:** Fully connected layers after the convolutional layers. These combine the spatial features extracted by conv layers into final predictions.

**How to set it:**
| Complexity | Suggested FC Layers |
|------------|-------------------|
| Simple | `[256]` |
| Medium | `[512, 256]` |
| Complex | `[1024, 512, 256]` |

**Guidelines:**
- First FC layer is typically larger (receives flattened conv output)
- Decrease size toward output (funnel shape)
- Too many neurons here can cause overfitting

---

#### `NUM_CLASSES`
```python
NUM_CLASSES = 10
```
**What it does:** The number of categories/classes to classify images into.

**How to set it:** Count your class folders. The script will warn you if this doesn't match and auto-correct.

| Task | NUM_CLASSES |
|------|-------------|
| Binary (cat vs dog) | `2` |
| Digits (0-9) | `10` |
| CIFAR-100 | `100` |

---

#### `DROPOUT`
```python
DROPOUT = 0.5
```
**What it does:** Randomly "drops out" (sets to zero) a percentage of neurons in the fully connected layers during training. This prevents overfitting.

**How to set it:**
| Situation | Suggested Dropout |
|-----------|-------------------|
| Small dataset, overfitting | 0.5 - 0.7 |
| Medium dataset | 0.3 - 0.5 |
| Large dataset | 0.2 - 0.3 |
| No overfitting | 0.0 |

**Note:** CNNs typically use higher dropout (0.5) in FC layers than MLPs because the FC layers have many parameters.

**Signs you need more dropout:** Training accuracy is much higher than test accuracy.

---

#### `ACTIVATION`
```python
ACTIVATION = 'relu'
```
**What it does:** The non-linear function applied after each convolutional and fully connected layer.

| Activation | Best For | Notes |
|------------|----------|-------|
| `'relu'` | General use (default) | Fast, works well for CNNs |
| `'leaky_relu'` | When ReLU neurons die | Allows small negative values |
| `'elu'` | Deeper networks | Smoother than ReLU |
| `'silu'` | Modern architectures | Used in EfficientNet |
| `'gelu'` | Transformer-style | Smoother, slightly slower |
| `'mish'` | Experimental | Self-regularizing |

**Recommendation:** `'relu'` is the standard choice for CNNs.

---

#### `BATCH_NORM`
```python
BATCH_NORM = True
```
**What it does:** Normalizes the outputs of each layer. This stabilizes training and often allows higher learning rates.

**When to use:**
| Situation | BATCH_NORM |
|-----------|------------|
| Almost always for CNNs | `True` |
| Very small batch sizes (<8) | `False` (stats unreliable) |
| Debugging/simplicity | `False` |

**Benefits:**
- Faster convergence
- Allows higher learning rates
- Acts as mild regularization
- Reduces sensitivity to initialization

**Recommendation:** Keep `True` for CNNs. It's standard practice.

---

### Data Configuration

#### `TRAIN_DATA_PATH` and `TEST_DATA_PATH`
```python
TRAIN_DATA_PATH = 'data/train'
TEST_DATA_PATH = 'data/test'
```
**What it does:** Paths to your training and test image folders.

**How to set it:** Use relative paths (from script location) or absolute paths.

---

#### `USE_DATA_AUGMENTATION`
```python
USE_DATA_AUGMENTATION = True
```
**What it does:** Applies random transformations to training images (flips, rotations, color changes) to artificially increase dataset diversity.

**Augmentations applied:**
- Random horizontal flip (50% chance)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)

**When to use:**
| Situation | USE_DATA_AUGMENTATION |
|-----------|----------------------|
| Small dataset | `True` (essential!) |
| Medium dataset | `True` (recommended) |
| Very large dataset | `True` or `False` |
| Testing/debugging | `False` (for reproducibility) |

**Benefits:** Dramatically reduces overfitting, especially with small datasets.

---

#### `NORMALIZE_IMAGENET`
```python
NORMALIZE_IMAGENET = True
```
**What it does:** Normalizes images using ImageNet statistics (mean and std values from millions of images).

**When to use:**
| Situation | NORMALIZE_IMAGENET |
|-----------|-------------------|
| Natural images (photos) | `True` |
| RGB images, transfer learning planned | `True` |
| Medical/satellite/specialized images | `False` (use `True` to try both) |
| Grayscale images | `False` (auto-disabled) |

**What it does internally:**
```python
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

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
| `'sgd'` | Fine-tuning, large datasets | Often used with learning rate scheduling |
| `'rmsprop'` | Non-stationary problems | Alternative to Adam |

**Recommendation:** Start with `'adam'`. Try `'sgd'` with momentum for potentially better final accuracy (but requires more tuning).

---

#### `LEARNING_RATE`
```python
LEARNING_RATE = 0.001
```
**What it does:** Controls how big of a step the optimizer takes when updating weights.

**How to set it:**
| Learning Rate | Effect |
|---------------|--------|
| Too high (>0.01) | Training unstable, accuracy oscillates |
| Good (0.0001-0.001) | Steady improvement |
| Too low (<0.0001) | Very slow training |

**Typical values:**
- Adam: `0.001` (good default)
- SGD: `0.01` with momentum `0.9`

**Tip:** If accuracy is very jumpy, reduce learning rate by 10x.

---

#### `BATCH_SIZE`
```python
BATCH_SIZE = 32
```
**What it does:** Number of images processed together before updating weights.

**How to set it:**
| Batch Size | Pros | Cons |
|------------|------|------|
| Small (8-16) | Better generalization, less memory | Slower, noisier gradients |
| Medium (32-64) | Good balance | — |
| Large (128+) | Faster training (with GPU) | May generalize worse, needs more memory |

**Memory guide:**
| Image Size | Typical Max Batch Size (8GB GPU) |
|------------|----------------------------------|
| 64×64 | 128-256 |
| 128×128 | 64-128 |
| 224×224 | 32-64 |
| 299×299 | 16-32 |

**Recommendation:** Start with `32`. Reduce if you get out-of-memory errors.

---

#### `EPOCHS`
```python
EPOCHS = 50
```
**What it does:** Number of times the model sees the entire training dataset.

**How to set it:**
- Watch the training plots
- Stop when accuracy plateaus
- If test accuracy drops while train accuracy rises → overfitting

**Typical values:**
| Dataset Size | Suggested Epochs |
|--------------|-----------------|
| Small (<1K images) | 50-100 |
| Medium (1K-10K) | 30-50 |
| Large (10K+) | 10-30 |

**Note:** With data augmentation, you can train longer since each epoch shows different variations.

---

## Tuning Guide

### Step-by-Step Approach

1. **Start simple:**
   ```python
   CONV_LAYERS = [32, 64]
   FC_LAYERS = [256]
   DROPOUT = 0.5
   BATCH_NORM = True
   LEARNING_RATE = 0.001
   USE_DATA_AUGMENTATION = True
   ```

2. **Get a baseline:** Run and note test accuracy.

3. **If underfitting** (both train and test accuracy are low):
   - Add more conv layers: `[32, 64, 128]`
   - Add more filters: `[64, 128, 256]`
   - Add more FC neurons: `[512, 256]`
   - Train for more epochs
   - Increase image size

4. **If overfitting** (train accuracy >> test accuracy):
   - Increase dropout: `DROPOUT = 0.6` or `0.7`
   - Reduce model size
   - Enable data augmentation
   - Get more training data
   - Reduce epochs

5. **Fine-tune:**
   - Adjust learning rate
   - Try different optimizers
   - Experiment with image size

### Architecture Scaling Guide

| Dataset Size | Suggested Architecture |
|--------------|----------------------|
| Tiny (<500 images) | `CONV_LAYERS = [16, 32]`, `FC_LAYERS = [128]`, `DROPOUT = 0.7` |
| Small (500-2K) | `CONV_LAYERS = [32, 64]`, `FC_LAYERS = [256]`, `DROPOUT = 0.5` |
| Medium (2K-10K) | `CONV_LAYERS = [32, 64, 128]`, `FC_LAYERS = [512, 256]`, `DROPOUT = 0.5` |
| Large (10K+) | `CONV_LAYERS = [64, 128, 256, 512]`, `FC_LAYERS = [1024, 512]`, `DROPOUT = 0.3` |

---

## Common Issues

### "Data folder not found" error
- Check that `TRAIN_DATA_PATH` and `TEST_DATA_PATH` exist
- Ensure folder structure follows ImageFolder format (class subfolders)
- Use forward slashes `/` even on Windows

### Training accuracy not improving
- Learning rate too low → increase `LEARNING_RATE`
- Model too simple → add more conv layers or filters
- Images too small → increase `INPUT_HEIGHT` and `INPUT_WIDTH`
- Check that images are actually different across classes

### Training accuracy high but test accuracy low (overfitting)
- Enable data augmentation → `USE_DATA_AUGMENTATION = True`
- Increase dropout → `DROPOUT = 0.6` or higher
- Reduce model size → fewer filters, fewer FC neurons
- Get more training data
- Reduce epochs

### Out of memory (CUDA OOM) error
- Reduce `BATCH_SIZE`
- Reduce image size (`INPUT_HEIGHT`, `INPUT_WIDTH`)
- Reduce `CONV_LAYERS` filter counts
- Reduce `FC_LAYERS` sizes

### Training is very slow
- Enable GPU (install CUDA version of PyTorch)
- Reduce image size
- Reduce model complexity
- Increase batch size (if memory allows)

### All predictions are the same class
- Check class balance (equal images per class?)
- Learning rate may be too high or too low
- Model may be too simple for the task
- Verify data is correctly labeled

### Low accuracy despite training well
- Image size may be too small to capture important details
- Need more convolutional layers for complex patterns
- Classes may be too visually similar
- Try different augmentation strategies

---

## Output Files

After running, the script produces:
- `cnn_model.pth` — Saved model weights
- `cnn_class_info.pth` — Class names and input configuration (for inference)
- `cnn_training_metrics.png` — Plots of training loss and accuracy

---

## Example Configurations

### Binary Classification (Cats vs Dogs)
```python
INPUT_CHANNELS = 3
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
CONV_LAYERS = [32, 64, 128]
FC_LAYERS = [512, 256]
NUM_CLASSES = 2
DROPOUT = 0.5
BATCH_NORM = True
USE_DATA_AUGMENTATION = True
LEARNING_RATE = 0.001
EPOCHS = 30
```

### Digit Recognition (MNIST-style)
```python
INPUT_CHANNELS = 1          # Grayscale
INPUT_HEIGHT = 28
INPUT_WIDTH = 28
CONV_LAYERS = [32, 64]
FC_LAYERS = [128]
NUM_CLASSES = 10
DROPOUT = 0.25
BATCH_NORM = True
USE_DATA_AUGMENTATION = False  # Digits shouldn't be flipped
NORMALIZE_IMAGENET = False
LEARNING_RATE = 0.001
EPOCHS = 20
```

### Medical Image Classification
```python
INPUT_CHANNELS = 3
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONV_LAYERS = [64, 128, 256, 512]
FC_LAYERS = [1024, 512]
NUM_CLASSES = 4
DROPOUT = 0.5
BATCH_NORM = True
USE_DATA_AUGMENTATION = True
NORMALIZE_IMAGENET = True
LEARNING_RATE = 0.0001      # Lower for sensitive tasks
EPOCHS = 50
```

### Small Dataset with Few Images
```python
INPUT_CHANNELS = 3
INPUT_HEIGHT = 64           # Smaller to reduce overfitting
INPUT_WIDTH = 64
CONV_LAYERS = [16, 32]      # Simple architecture
FC_LAYERS = [64]
NUM_CLASSES = 3
DROPOUT = 0.7               # High dropout
BATCH_NORM = True
USE_DATA_AUGMENTATION = True  # Essential!
LEARNING_RATE = 0.001
EPOCHS = 100                # More epochs with augmentation
```

### High-Resolution Complex Classification
```python
INPUT_CHANNELS = 3
INPUT_HEIGHT = 299
INPUT_WIDTH = 299
CONV_LAYERS = [64, 128, 256, 512, 512]
FC_LAYERS = [1024, 512, 256]
NUM_CLASSES = 100
DROPOUT = 0.5
BATCH_NORM = True
USE_DATA_AUGMENTATION = True
OPTIMIZER = 'adamw'
LEARNING_RATE = 0.0005
BATCH_SIZE = 16             # Smaller due to large images
EPOCHS = 50
```

---

## Loading a Saved Model

To use your trained model later:

```python
import torch
from torchvision import transforms
from PIL import Image

# Load class info
class_info = torch.load('cnn_class_info.pth')
class_names = class_info['class_names']

# Recreate the model (must match training config)
model = CNNModel(
    input_channels=class_info['input_channels'],
    input_height=class_info['input_height'],
    input_width=class_info['input_width'],
    num_classes=len(class_names)
)

# Load weights
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((class_info['input_height'], class_info['input_width'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('test_image.jpg')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    
print(f"Predicted class: {class_names[predicted.item()]}")
```

---

## Tips for Better Results

1. **Balance your classes** — Equal images per class prevents bias
2. **Use data augmentation** — Essential for small datasets
3. **Start small, scale up** — Begin with simple architecture and add complexity
4. **Monitor both metrics** — Watch train AND test accuracy
5. **Save frequently** — Training can be interrupted; save checkpoints
6. **Visualize failures** — Look at misclassified images to understand weaknesses
7. **Consider transfer learning** — For complex tasks, pretrained models often work better
