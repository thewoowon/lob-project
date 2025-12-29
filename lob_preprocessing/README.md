# LOB Preprocessing: Systematic Comparison Study

A comprehensive framework for comparing data preprocessing methods in Limit Order Book (LOB) mid-price prediction.

## Overview

This project systematically compares various preprocessing methods (Savitzky-Golay, Kalman Filter, Wavelet Denoising, Moving Average) with different machine learning models (Logistic Regression, XGBoost, CatBoost, CNN, DeepLOB, CNN-LSTM) for LOB mid-price prediction.

**Key Research Question**: *Does data preprocessing matter more than model complexity in LOB prediction?*

## Features

- **5 Preprocessing Methods**: Raw, Savitzky-Golay, Kalman Filter, Wavelet Denoising, Moving Average
- **6 ML Models**: Logistic Regression, XGBoost, CatBoost, CNN, DeepLOB, CNN-LSTM
- **Multiple Configurations**: LOB depths (5, 10, 20, 40 levels), prediction horizons (100ms - 10s)
- **Comprehensive Evaluation**: Accuracy, F1, MCC, SNR, computational metrics
- **Automated Pipeline**: Data generation/download → preprocessing → feature extraction → training → evaluation
- **Visualization Tools**: Heatmaps, comparison plots, confusion matrices

## Project Structure

```
lob_preprocessing/
├── data/
│   ├── download.py          # Data download (Bybit, Binance, synthetic)
│   ├── preprocess.py         # Preprocessing methods
│   └── features.py           # Feature engineering
├── models/
│   ├── baseline.py           # Traditional ML models
│   └── deep_models.py        # Deep learning models
├── experiments/
│   └── run_experiments.py    # Main experiment runner
├── analysis/
│   ├── evaluate.py           # Evaluation metrics
│   └── visualize.py          # Visualization
├── configs/
│   └── config.yaml           # Configuration file
├── notebooks/
│   ├── EDA.ipynb
│   └── Results.ipynb
└── utils.py                  # Utility functions
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd lob-project
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install PyTorch with GPU support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Run Quick Test

Test the framework with minimal configurations:

```bash
cd lob_preprocessing
python experiments/run_experiments.py --quick
```

This runs:
- 2 preprocessing methods (raw, savgol)
- 2 models (logistic, xgboost)
- 1 depth (10 levels)
- 1 horizon (1000ms)

### 2. Run Full Experiment Suite

```bash
python experiments/run_experiments.py
```

This runs all combinations (may take hours):
- 5 preprocessing methods
- 3 models (configurable)
- 4 depths
- 5 horizons

### 3. Analyze Results

```bash
python experiments/run_experiments.py --analyze
```

## Usage Examples

### Generate Synthetic Data

```python
from data.download import SyntheticLOBGenerator

generator = SyntheticLOBGenerator(seed=42)
df = generator.generate(n_snapshots=10000, depth=10)
df.to_csv('data/raw/synthetic_lob.csv', index=False)
```

### Apply Preprocessing

```python
from data.preprocess import LOBPreprocessor
import numpy as np

# Create preprocessor
preprocessor = LOBPreprocessor(method='savgol', window_length=11, polyorder=2)

# Apply to data
mid_prices = np.array([...])
filtered = preprocessor.fit_transform(mid_prices)

# Compute SNR
snr = preprocessor.compute_snr(mid_prices, filtered)
print(f"SNR: {snr:.2f} dB")
```

### Extract Features

```python
from data.features import LOBFeatureEngineer, create_lob_dataset
import pandas as pd

# Load LOB data
df = pd.read_csv('data/raw/synthetic_lob.csv')

# Create dataset
X, y = create_lob_dataset(
    df,
    depth=10,
    horizon_ms=1000,
    task='ternary',
    threshold=0.5
)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### Train Models

```python
from models.baseline import get_baseline_model
from models.deep_models import get_deep_model

# Baseline model
model = get_baseline_model('xgboost')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Deep learning model
trainer = get_deep_model('cnn', input_size=X_train.shape[1], num_classes=3)
trainer.fit(X_train, y_train, epochs=50)
y_pred = trainer.predict(X_test)
```

### Evaluate and Visualize

```python
from analysis.evaluate import ModelEvaluator
from analysis.visualize import ResultsVisualizer
import pandas as pd

# Evaluate
evaluator = ModelEvaluator(num_classes=3)
metrics = evaluator.evaluate(y_test, y_pred)
evaluator.print_report(metrics)

# Visualize results
results_df = pd.read_csv('results/experiment_results.csv')
viz = ResultsVisualizer(results_df)
viz.generate_all_plots()
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Data settings**: Assets, periods, LOB depths
- **Preprocessing parameters**: Window sizes, filter parameters
- **Model hyperparameters**: Learning rates, epochs, etc.
- **Experiment settings**: Train/val/test split, random seed

Example:

```yaml
data:
  lob_depths: [5, 10, 20, 40]

preprocessing:
  methods:
    - raw
    - savgol
    - kalman

models:
  xgboost:
    max_depth: 3
    learning_rate: 0.1
    n_estimators: 100
```

## Expected Results

Based on recent literature and preliminary experiments:

1. **Preprocessing helps**: 1-2% improvement over raw data
2. **Simple models competitive**: XGBoost + preprocessing ≈ DeepLOB (raw)
3. **Method matters**: Savitzky-Golay best for speed/accuracy trade-off
4. **Horizon dependency**: Short horizons benefit more from preprocessing

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test preprocessing
python data/preprocess.py

# Test feature engineering
python data/features.py

# Test baseline models
python models/baseline.py

# Test deep learning models
python models/deep_models.py

# Test evaluation
python analysis/evaluate.py

# Test visualization
python analysis/visualize.py
```

## Data Sources

### Cryptocurrency (Free)

- **Bybit**: https://www.bybit.com/derivatives/en/history-data
- **Binance**: https://data.binance.vision/

### Equity (Paid/Academic)

- **LOBSTER**: https://lobsterdata.com/ (NASDAQ)
- **KOSCOM**: Korean market data (academic license)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lob_preprocessing_2024,
  title={Systematic Comparison of Preprocessing Methods for LOB Mid-Price Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lob-preprocessing}
}
```

## References

1. Zhang et al. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
2. Recent paper (2025). "Exploring Microstructural Dynamics: Cryptocurrency LOB Analysis"

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

- Author: Your Name
- Email: your.email@domain.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Status**: 🚧 Research in Progress

Last updated: 2024-12-04
