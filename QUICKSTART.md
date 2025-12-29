# Quick Start Guide

Get started with LOB preprocessing experiments in 5 minutes!

## Step 1: Installation (2 min)

```bash
# Clone repository
cd lob-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Quick Test (1 min)

Run a minimal experiment to test everything works:

```bash
cd lob_preprocessing
python experiments/run_experiments.py --quick
```

Expected output:
```
Generating synthetic LOB data...
Generated data shape: (10000, 43)
Running experiments: 100%|████████████| 4/4 [00:30<00:00,  7.5s/it]
Results saved to: results/experiment_results.csv

RESULTS SUMMARY
Best configuration (by accuracy):
  Preprocessing: savgol
  Model: xgboost
  Accuracy: 0.5234
```

## Step 3: View Results (1 min)

```bash
# Analyze results
python experiments/run_experiments.py --analyze

# Results are in:
# - results/experiment_results.csv
# - results/plots/*.png
```

## Step 4: Customize Configuration (1 min)

Edit `configs/config.yaml`:

```yaml
# Change preprocessing methods
preprocessing:
  methods:
    - raw
    - savgol
    - kalman

# Change models to test
# (comment out deep learning models for faster testing)
```

## Step 5: Run Full Experiments

```bash
# This will run all combinations (may take 1-2 hours)
python experiments/run_experiments.py
```

---

## Common Issues

### Issue: Import errors

**Solution**: Make sure you're in the correct directory and virtual environment is activated

```bash
cd lob-project/lob_preprocessing
source ../venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Issue: GPU not detected

**Solution**: Install PyTorch with CUDA support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or set `use_gpu: False` in `configs/config.yaml`

### Issue: Out of memory

**Solution**: Reduce batch size or number of samples

```yaml
models:
  cnn:
    batch_size: 32  # Reduce from 64
```

---

## Next Steps

1. **Generate real data**: Replace synthetic data with Bybit/Binance data
2. **Add more models**: Implement your own model in `models/`
3. **Custom features**: Add domain-specific features in `data/features.py`
4. **Deep analysis**: Use Jupyter notebooks in `notebooks/`

## Example Jupyter Notebook

Create `notebooks/quick_analysis.ipynb`:

```python
import pandas as pd
import sys
sys.path.append('..')

from analysis.visualize import ResultsVisualizer

# Load results
df = pd.read_csv('../results/experiment_results.csv')

# Visualize
viz = ResultsVisualizer(df)
viz.generate_all_plots()

# Best configuration
print(df.nlargest(10, 'accuracy')[['preprocess', 'model', 'accuracy']])
```

---

## Getting Help

- Read full documentation: [README.md](README.md)
- Check module tests: `python data/preprocess.py`
- Review config options: [configs/config.yaml](configs/config.yaml)

Happy experimenting! 🚀
