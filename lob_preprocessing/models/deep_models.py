"""
Deep Learning Models for LOB Mid-Price Prediction

Implements:
1. Simple CNN
2. DeepLOB (Zhang et al. 2019)
3. CNN + LSTM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional
import time


class SimpleCNN(nn.Module):
    """Simple CNN for LOB prediction"""

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        filters: list = None,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super(SimpleCNN, self).__init__()

        if filters is None:
            filters = [64, 64, 32]

        layers = []

        # First conv block
        layers.extend([
            nn.Conv1d(1, filters[0], kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(filters[0]),
            nn.Dropout(dropout)
        ])

        # Additional conv blocks
        for i in range(1, len(filters)):
            layers.extend([
                nn.Conv1d(filters[i-1], filters[i], kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(filters[i]),
                nn.Dropout(dropout)
            ])

        self.conv_layers = nn.Sequential(*layers)

        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(filters[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Conv layers
        x = self.conv_layers(x)

        # Global pooling: (batch, channels, length) -> (batch, channels, 1)
        x = self.global_pool(x)

        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)

        # FC layers
        x = self.fc(x)

        return x


class DeepLOB(nn.Module):
    """
    DeepLOB model from Zhang et al. 2019
    "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        conv_filters: int = 32,
        lstm_units: int = 64
    ):
        super(DeepLOB, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(conv_filters),
        )

        # Inception module
        self.inception1 = nn.Sequential(
            nn.Conv1d(conv_filters, 64, kernel_size=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        # x shape: (batch, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # Convolutional layers with residual connections
        h1 = self.conv1(x)
        h2 = self.conv2(h1) + h1  # Residual
        h3 = self.conv3(h2) + h2  # Residual

        # Inception
        h4 = self.inception1(h3)

        # Reshape for LSTM: (batch, channels, length) -> (batch, length, channels)
        h4 = h4.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(h4)

        # Take last output
        lstm_out = lstm_out[:, -1, :]

        # FC
        out = self.fc(lstm_out)

        return out


class CNNLSTM(nn.Module):
    """CNN + LSTM hybrid model"""

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        conv_filters: int = 64,
        lstm_units: list = None,
        dropout: float = 0.2
    ):
        super(CNNLSTM, self).__init__()

        if lstm_units is None:
            lstm_units = [64, 32]

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters),
            nn.Dropout(dropout),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_filters),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units[0],
            num_layers=len(lstm_units),
            batch_first=True,
            dropout=dropout if len(lstm_units) > 1 else 0
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_units[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # CNN
        x = self.conv(x)

        # Reshape for LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last output
        lstm_out = lstm_out[:, -1, :]

        # FC
        out = self.fc(lstm_out)

        return out


class DeepModelTrainer:
    """Trainer for deep learning models"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 64
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.training_time = 0
        self.inference_time = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        verbose: bool = True
    ):
        """Train the model"""
        start_time = time.time()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                train_acc = 100 * correct / total
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.2f}%")

                # Validation
                if X_val is not None and y_val is not None:
                    val_acc = self.evaluate(X_val, y_val)
                    print(f"  Val Acc: {val_acc:.2f}%")

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Batch prediction
            predictions = []
            batch_size = self.batch_size

            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        self.inference_time = (time.time() - start_time) / len(X)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            probas = []
            batch_size = self.batch_size

            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1)
                probas.extend(proba.cpu().numpy())

        return np.array(probas)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy"""
        predictions = self.predict(X)
        accuracy = 100 * np.mean(predictions == y)
        return accuracy


def get_deep_model(
    model_name: str,
    input_size: int,
    num_classes: int = 3,
    device: str = 'cpu',
    **kwargs
) -> DeepModelTrainer:
    """
    Factory function to get deep learning model

    Args:
        model_name: One of ['cnn', 'deeplob', 'cnn_lstm']
        input_size: Number of input features
        num_classes: Number of output classes
        device: 'cpu' or 'cuda'
        **kwargs: Model-specific parameters

    Returns:
        DeepModelTrainer instance
    """
    models = {
        'cnn': SimpleCNN,
        'deeplob': DeepLOB,
        'cnn_lstm': CNNLSTM,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    # Create model
    model_class = models[model_name.lower()]
    model = model_class(input_size=input_size, num_classes=num_classes, **kwargs)

    # Create trainer
    trainer = DeepModelTrainer(
        model=model,
        device=device,
        learning_rate=kwargs.get('learning_rate', 0.001),
        batch_size=kwargs.get('batch_size', 64)
    )

    return trainer


def main():
    """Example usage"""
    print("=== Deep Learning Models Module ===\n")

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, 3, n_samples)

    X_test = np.random.randn(200, n_features).astype(np.float32)
    y_test = np.random.randint(0, 3, 200)

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}\n")

    # Test all models
    models = ['cnn', 'deeplob', 'cnn_lstm']

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_name.upper()}")
        print('='*50)

        try:
            # Initialize model
            trainer = get_deep_model(
                model_name,
                input_size=n_features,
                num_classes=3,
                device=device,
                batch_size=32
            )

            # Train
            print("Training...")
            trainer.fit(X_train, y_train, epochs=20, verbose=False)
            print(f"  Training time: {trainer.training_time:.3f}s")

            # Predict
            print("Predicting...")
            y_pred = trainer.predict(X_test)
            print(f"  Inference time: {trainer.inference_time*1000:.3f}ms per sample")

            # Accuracy
            accuracy = np.mean(y_pred == y_test)
            print(f"  Accuracy: {accuracy:.4f}")

            print(f"✓ {model_name} test passed!")

        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")

    print("\n\n✓ All deep learning models tested!")


if __name__ == "__main__":
    main()
