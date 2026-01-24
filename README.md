# ML Templates

A collection of beginner-friendly PyTorch templates for creating and training neural networks. Each template features easy configuration with all settings in one place at the top of the file—just set your parameters and run!

---

## Available Templates

| Template | Best For |
|----------|----------|
| [EasyNN](#easynn) | Tabular data, classification, regression |
| [EasyRNN](#easyrnn) | Time series, sequences, forecasting |
| [EasyCNN](#easycnn) | Image data, spatial patterns |

---

## EasyNN

**Multilayer Perceptron (MLP) for tabular/structured data**

Use EasyNN when you have CSV data with independent samples—each row is a separate data point with features as columns. Great for classification and regression tasks.

**Best for:**
- Classification (spam detection, disease diagnosis)
- Regression (price prediction, score estimation)
- Any tabular dataset with extracted features

📖 **[Full Documentation →](README_EasyNN.md)**

---

## EasyRNN

**Recurrent Neural Network for sequential data**

Use EasyRNN when the order of your data matters. Ideal for time series forecasting and sequence prediction where past values influence future predictions.

**Best for:**
- Time series forecasting (stock prices, weather, sensors)
- Predictive maintenance (Remaining Useful Life prediction)
- Sequence classification (activity recognition, anomaly detection)

📖 **[Full Documentation →](README_EasyRNN.md)**

---

## EasyCNN

**Convolutional Neural Network for image and spatial data**

Use EasyCNN when working with image data or any data where spatial relationships between neighboring elements matter.

**Will support:**
- Image classification
- Pattern recognition
- Feature extraction from visual data

---

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

---

## Quick Start

1. **Choose your template** based on your data type (see table above)
2. **Prepare your data** as a CSV file
3. **Configure parameters** at the top of the script
4. **Run the script**: `python EasyNN.py` or `python EasyRNN.py`
5. **Review results** in the console output and generated plots

---
## Credits

- Originally developed following [Codemy.com PyTorch tutorials](https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&si=qN9cA47jvIpXJ6Ez)
- Code assistance and ReadMe generation provided by Claude Opus 4.5 (Anthropic), an AI assistant

## License

MIT License - Feel free to use, modify, and distribute.
