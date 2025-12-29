"""
TransLOB Fair Comparison

Purpose: Compare TransLOB with our approach using same feature dimensionality

Configurations:
1. TransLOB (raw 40) - baseline from paper
2. TransLOB (raw 40 + engineered 38) - our features with TransLOB
3. CatBoost (raw 40) - our baseline
4. CatBoost (raw 40 + engineered 38) - our approach

This provides fair comparison:
- Same data (FI-2010)
- Same features when applicable
- Multiple seeds for statistical validation

Expected insight:
- Does TransLOB benefit from engineered features?
- How much does model architecture matter?
- Is our improvement mainly from features or model?
"""

import sys
sys.path.insert(0, '/Users/aepeul/lob-project/lob_preprocessing')

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data.fi2010_loader import FI2010Loader
from data.feature_engineering import LOBFeatureEngineering
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import time


# ============================================================
# TransLOB Model Implementation
# ============================================================

class TransLOBDataset(Dataset):
    """Dataset for TransLOB"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransLOB(nn.Module):
    """
    Simplified TransLOB implementation based on:
    Wallbridge (2020) "Transformers for Limit Order Books"

    Architecture:
    - Input: LOB features
    - Transformer encoder layers
    - Classification head
    """

    def __init__(self, input_dim, num_classes=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransLOB, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (simple learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim)

        # Project to d_model
        x = self.input_proj(x)  # (batch, d_model)

        # Add batch and sequence dimensions for transformer
        x = x.unsqueeze(1)  # (batch, 1, d_model)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer
        x = self.transformer(x)  # (batch, 1, d_model)

        # Take the output
        x = x.squeeze(1)  # (batch, d_model)

        # Classification
        out = self.fc(x)  # (batch, num_classes)

        return out


def train_translob(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train TransLOB model"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        val_acc = 100. * val_correct / val_total

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return model


def evaluate_translob(model, test_loader, device='cpu'):
    """Evaluate TransLOB model"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# ============================================================
# Comparison Experiments
# ============================================================

def run_comparison(seed=42):
    """Run all 4 configurations with one seed"""

    print(f"\n{'='*70}")
    print(f"Seed {seed}")
    print(f"{'='*70}\n")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load data
    print("Loading data...")
    loader = FI2010Loader(normalization='zscore', auction=False)
    X_train_raw, y_train = loader.load_all_training(horizon=100, days=[1, 2])
    X_test_raw, y_test = loader.load_test(horizon=100)

    print(f"Train: {len(X_train_raw):,}, Test: {len(X_test_raw):,}")

    # Extract engineered features
    print("Extracting engineered features...")
    fe = LOBFeatureEngineering(depth=10)
    X_train_eng, _ = fe.extract_all_features(X_train_raw, include_raw=False)
    X_test_eng, _ = fe.extract_all_features(X_test_raw, include_raw=False)

    # Combine
    X_train_combined = np.hstack([X_train_raw, X_train_eng])
    X_test_combined = np.hstack([X_test_raw, X_test_eng])

    print(f"Raw: {X_train_raw.shape[1]}, Engineered: {X_train_eng.shape[1]}, Combined: {X_train_combined.shape[1]}\n")

    results = {}

    # ========================================
    # Config 1: TransLOB (raw 40)
    # ========================================
    print("="*70)
    print("Config 1: TransLOB (raw 40)")
    print("="*70)

    # Create datasets
    train_dataset = TransLOBDataset(X_train_raw, y_train)
    test_dataset = TransLOBDataset(X_test_raw, y_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create model
    model = TransLOB(input_dim=40, num_classes=3, d_model=64, nhead=4, num_layers=2)

    # Train
    print("Training...")
    start = time.time()
    model = train_translob(model, train_loader, test_loader, epochs=50, lr=0.001, device=device)
    train_time = time.time() - start

    # Evaluate
    print("Evaluating...")
    acc = evaluate_translob(model, test_loader, device=device)

    results['translob_raw'] = {'accuracy': acc, 'train_time': train_time}
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"Training time: {train_time:.1f}s\n")

    # ========================================
    # Config 2: TransLOB (raw + engineered 78)
    # ========================================
    print("="*70)
    print("Config 2: TransLOB (raw 40 + engineered 38)")
    print("="*70)

    # Create datasets
    train_dataset = TransLOBDataset(X_train_combined, y_train)
    test_dataset = TransLOBDataset(X_test_combined, y_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create model (larger input)
    model = TransLOB(input_dim=78, num_classes=3, d_model=64, nhead=4, num_layers=2)

    # Train
    print("Training...")
    start = time.time()
    model = train_translob(model, train_loader, test_loader, epochs=50, lr=0.001, device=device)
    train_time = time.time() - start

    # Evaluate
    print("Evaluating...")
    acc = evaluate_translob(model, test_loader, device=device)

    results['translob_combined'] = {'accuracy': acc, 'train_time': train_time}
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"Training time: {train_time:.1f}s\n")

    # ========================================
    # Config 3: CatBoost (raw 40)
    # ========================================
    print("="*70)
    print("Config 3: CatBoost (raw 40)")
    print("="*70)

    print("Training...")
    start = time.time()
    model = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=seed,
        verbose=False
    )
    model.fit(X_train_raw, y_train)
    train_time = time.time() - start

    print("Evaluating...")
    y_pred = model.predict(X_test_raw)
    acc = accuracy_score(y_test, y_pred)

    results['catboost_raw'] = {'accuracy': acc, 'train_time': train_time}
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"Training time: {train_time:.1f}s\n")

    # ========================================
    # Config 4: CatBoost (raw + engineered 78)
    # ========================================
    print("="*70)
    print("Config 4: CatBoost (raw 40 + engineered 38)")
    print("="*70)

    print("Training...")
    start = time.time()
    model = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=seed,
        verbose=False
    )
    model.fit(X_train_combined, y_train)
    train_time = time.time() - start

    print("Evaluating...")
    y_pred = model.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)

    results['catboost_combined'] = {'accuracy': acc, 'train_time': train_time}
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"Training time: {train_time:.1f}s\n")

    return results


def main():
    """Run comparison with multiple seeds"""

    print("\n" + "="*70)
    print("TransLOB vs CatBoost Fair Comparison")
    print("="*70)

    seeds = [42, 123, 456]  # 3 seeds
    all_results = []

    for seed in seeds:
        results = run_comparison(seed)
        results['seed'] = seed
        all_results.append(results)

    # Aggregate
    print("\n" + "="*70)
    print("FINAL RESULTS (averaged over seeds)")
    print("="*70)
    print()

    df_data = []
    for config_name in ['translob_raw', 'translob_combined', 'catboost_raw', 'catboost_combined']:
        accs = [r[config_name]['accuracy'] for r in all_results]
        times = [r[config_name]['train_time'] for r in all_results]

        df_data.append({
            'Configuration': config_name,
            'Accuracy': np.mean(accs) * 100,
            'Std': np.std(accs, ddof=1) * 100,
            'Train Time (s)': np.mean(times)
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    translob_raw = df[df['Configuration'] == 'translob_raw']['Accuracy'].values[0]
    translob_comb = df[df['Configuration'] == 'translob_combined']['Accuracy'].values[0]
    catboost_raw = df[df['Configuration'] == 'catboost_raw']['Accuracy'].values[0]
    catboost_comb = df[df['Configuration'] == 'catboost_combined']['Accuracy'].values[0]

    print(f"Feature engineering benefit:")
    print(f"  TransLOB: {translob_comb - translob_raw:+.2f} pp")
    print(f"  CatBoost: {catboost_comb - catboost_raw:+.2f} pp")
    print()

    print(f"Model architecture effect:")
    print(f"  Raw features:      CatBoost vs TransLOB = {catboost_raw - translob_raw:+.2f} pp")
    print(f"  Combined features: CatBoost vs TransLOB = {catboost_comb - translob_comb:+.2f} pp")
    print()

    # Save
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'translob_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return df


if __name__ == "__main__":
    main()
