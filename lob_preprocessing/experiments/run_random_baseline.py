"""
Random Feature Baseline Experiment

Purpose: Isolate the effect of feature dimensionality from domain knowledge
         by comparing engineered features with random features.

Experiment:
- Raw only (40 features): baseline
- Raw + Random (40 + 38 = 78 features): dimensionality effect
- Raw + Engineered (40 + 38 = 78 features): domain knowledge effect

Expected:
- Random should show minimal improvement (feature count effect only)
- Engineered should show significant improvement (domain knowledge)
- Delta between them = pure domain knowledge contribution
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

from data.fi2010_loader import FI2010Loader
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import time

def generate_random_features(n_samples, n_features=38, seed=42):
    """
    Generate random features with similar statistical properties to engineered features.

    Strategy:
    - Use different distributions (normal, uniform, exponential)
    - Add correlations between some features
    - Ensure numerical stability
    """
    np.random.seed(seed)

    random_features = []

    # Group 1: Normal distribution (like OI, OFI)
    normal_features = np.random.randn(n_samples, 12) * 0.5
    random_features.append(normal_features)

    # Group 2: Uniform distribution (like ratios)
    uniform_features = np.random.uniform(-1, 1, (n_samples, 10))
    random_features.append(uniform_features)

    # Group 3: Exponential-like (like volumes, impacts)
    exp_features = np.random.exponential(0.3, (n_samples, 10))
    random_features.append(exp_features)

    # Group 4: Correlated features (some random features correlate with each other)
    base = np.random.randn(n_samples, 3)
    corr_features = np.column_stack([
        base[:, 0],
        base[:, 0] * 0.7 + np.random.randn(n_samples) * 0.3,
        base[:, 1],
        base[:, 1] * 0.5 + base[:, 2] * 0.5,
        base[:, 2],
        np.random.randn(n_samples)
    ])
    random_features.append(corr_features)

    # Concatenate all
    random_features = np.hstack(random_features)

    # Ensure exactly 38 features
    if random_features.shape[1] > n_features:
        random_features = random_features[:, :n_features]
    elif random_features.shape[1] < n_features:
        extra = n_features - random_features.shape[1]
        random_features = np.hstack([
            random_features,
            np.random.randn(n_samples, extra)
        ])

    return random_features


def run_random_baseline_experiment(random_seed=42):
    """
    Run single random baseline experiment with one random seed.
    """
    print(f"\n{'='*60}")
    print(f"Random Baseline Experiment (seed={random_seed})")
    print(f"{'='*60}\n")

    # Load data
    print("Loading FI-2010 data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test, y_test = loader.load_test(horizon=100)

    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Raw features: {X_train.shape[1]}")

    # Generate random features
    print("\nGenerating random features...")
    random_train = generate_random_features(len(X_train), n_features=38, seed=random_seed)
    random_test = generate_random_features(len(X_test), n_features=38, seed=random_seed)

    print(f"Random features shape: {random_train.shape}")

    # Combine raw + random
    X_train_random = np.hstack([X_train, random_train])
    X_test_random = np.hstack([X_test, random_test])

    print(f"Combined features (raw + random): {X_train_random.shape[1]}")

    # Train model
    print("\nTraining CatBoost model...")
    model = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=random_seed,
        verbose=False
    )

    start_time = time.time()
    model.fit(X_train_random, y_train)
    train_time = time.time() - start_time

    # Predict
    print("Predicting...")
    y_pred = model.predict(X_test_random)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"Results (seed={random_seed}):")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Training time: {train_time:.1f}s")
    print(f"{'='*60}\n")

    return accuracy


def run_multiple_seeds():
    """
    Run random baseline with 5 seeds for statistical validation.
    """
    seeds = [42, 123, 456, 789, 1011]
    accuracies = []

    print("\n" + "="*60)
    print("RANDOM FEATURE BASELINE - MULTI-SEED VALIDATION")
    print("="*60)

    for seed in seeds:
        acc = run_random_baseline_experiment(random_seed=seed)
        accuracies.append(acc)

    # Compute statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    print("\n" + "="*60)
    print("FINAL RESULTS - RANDOM BASELINE")
    print("="*60)
    print(f"\nSeeds: {seeds}")
    print(f"Accuracies: {[f'{a*100:.2f}%' for a in accuracies]}")
    print(f"\nMean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Min: {min(accuracies)*100:.2f}%")
    print(f"Max: {max(accuracies)*100:.2f}%")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame({
        'seed': seeds,
        'accuracy': accuracies
    })

    output_file = results_dir / 'random_baseline_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Compare with baseline and engineered
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER CONFIGURATIONS")
    print("="*60)
    print("\nConfiguration          | Accuracy      | Δ vs Raw  | Features")
    print("-" * 70)
    print(f"Raw baseline           | 62.61% ± 0.36 | -         | 40")
    print(f"Raw + Random           | {mean_acc*100:.2f}% ± {std_acc*100:.2f} | {(mean_acc-0.6261)*100:+.2f} pp | 78")
    print(f"Raw + Engineered       | 68.90% ± 0.12 | +6.29 pp  | 78")
    print("-" * 70)

    # Domain knowledge contribution
    domain_contribution = 0.6890 - mean_acc
    print(f"\nDomain knowledge contribution: {domain_contribution*100:+.2f} pp")
    print(f"(Engineered - Random = pure domain knowledge effect)")

    # Statistical test (informal - we'd need paired test properly)
    print(f"\nInterpretation:")
    if mean_acc < 0.64:
        print("✅ Random features provide minimal benefit (<1.5 pp)")
        print("✅ Engineered features provide substantial benefit (6+ pp)")
        print("✅ Domain knowledge contributes ~5+ pp beyond dimensionality")
    else:
        print("⚠️ Random features show unexpected improvement (>1.5 pp)")
        print("⚠️ Need to investigate why")

    print("\n" + "="*60 + "\n")

    return results_df


if __name__ == "__main__":
    # Run multi-seed validation
    results = run_multiple_seeds()
