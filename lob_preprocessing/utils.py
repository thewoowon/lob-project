"""
Utility functions for LOB preprocessing project
"""

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import json


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("lob_preprocessing")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/lob_preprocessing.log")
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def ensure_dir(directory: str) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def compute_mid_price(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    """Compute mid-price from bid and ask prices"""
    return (bid_price + ask_price) / 2


def compute_spread(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    """Compute bid-ask spread"""
    return ask_price - bid_price


def compute_microprice(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    bid_volume: np.ndarray,
    ask_volume: np.ndarray
) -> np.ndarray:
    """
    Compute volume-weighted microprice
    microprice = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
    """
    total_volume = bid_volume + ask_volume
    # Avoid division by zero
    total_volume = np.where(total_volume == 0, 1, total_volume)
    microprice = (bid_price * ask_volume + ask_price * bid_volume) / total_volume
    return microprice


def compute_order_imbalance(bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
    """
    Compute order imbalance
    OI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    """
    total_volume = bid_volume + ask_volume
    # Avoid division by zero
    total_volume = np.where(total_volume == 0, 1, total_volume)
    order_imbalance = (bid_volume - ask_volume) / total_volume
    return order_imbalance


def label_price_movement(
    current_price: np.ndarray,
    future_price: np.ndarray,
    threshold: float = 0.0,
    task: str = "binary"
) -> np.ndarray:
    """
    Label price movements for prediction

    Args:
        current_price: Current mid-price
        future_price: Future mid-price
        threshold: Threshold for ternary classification (in price units)
        task: 'binary' or 'ternary'

    Returns:
        labels: 0 (down), 1 (stationary/up for binary), 2 (up for ternary)
    """
    price_change = future_price - current_price

    if task == "binary":
        # 0: down, 1: up
        labels = (price_change > 0).astype(int)
    elif task == "ternary":
        # 0: down, 1: stationary, 2: up
        labels = np.zeros_like(price_change, dtype=int)
        labels[price_change > threshold] = 2  # Up
        labels[np.abs(price_change) <= threshold] = 1  # Stationary
        labels[price_change < -threshold] = 0  # Down
    else:
        raise ValueError(f"Unknown task: {task}")

    return labels


def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> tuple:
    """
    Split data temporally (no shuffling to avoid look-ahead bias)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_device():
    """Get torch device (GPU if available)"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    except ImportError:
        print("PyTorch not installed, defaulting to CPU")
        return "cpu"


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f} seconds")

    def __str__(self):
        if self.elapsed is not None:
            return f"{self.elapsed:.4f}s"
        return "Timer not finished"
