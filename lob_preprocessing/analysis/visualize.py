"""
Visualization module for LOB preprocessing experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List


class ResultsVisualizer:
    """Visualize experiment results"""

    def __init__(self, results_df: pd.DataFrame, output_dir: str = "results/plots"):
        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_preprocessing_comparison(
        self,
        metric: str = 'accuracy',
        save: bool = True
    ):
        """
        Compare preprocessing methods across all experiments

        Args:
            metric: Metric to plot (accuracy, f1_macro, etc.)
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by preprocessing method
        grouped = self.results_df.groupby('preprocess')[metric].agg(['mean', 'std'])
        grouped = grouped.sort_values('mean', ascending=False)

        x = np.arange(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Preprocessing Method Comparison ({metric})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f'preprocessing_comparison_{metric}.png', dpi=300)
            print(f"Saved: preprocessing_comparison_{metric}.png")

        plt.close()

    def plot_model_comparison(
        self,
        metric: str = 'accuracy',
        save: bool = True
    ):
        """Compare models across all experiments"""
        fig, ax = plt.subplots(figsize=(12, 6))

        grouped = self.results_df.groupby('model')[metric].agg(['mean', 'std'])
        grouped = grouped.sort_values('mean', ascending=False)

        x = np.arange(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Model Comparison ({metric})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f'model_comparison_{metric}.png', dpi=300)
            print(f"Saved: model_comparison_{metric}.png")

        plt.close()

    def plot_heatmap(
        self,
        metric: str = 'accuracy',
        fixed_depth: int = 10,
        fixed_horizon: int = 1000,
        save: bool = True
    ):
        """
        Plot heatmap of preprocessing × model performance

        Args:
            metric: Metric to visualize
            fixed_depth: Fix depth to this value
            fixed_horizon: Fix horizon to this value
            save: Whether to save
        """
        # Filter data
        filtered = self.results_df[
            (self.results_df['depth'] == fixed_depth) &
            (self.results_df['horizon_ms'] == fixed_horizon)
        ]

        if len(filtered) == 0:
            print(f"No data for depth={fixed_depth}, horizon={fixed_horizon}")
            return

        # Pivot table
        pivot = filtered.pivot_table(
            values=metric,
            index='preprocess',
            columns='model',
            aggfunc='mean'
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': metric.capitalize()}
        )
        ax.set_title(
            f'{metric.capitalize()} Heatmap\n'
            f'(Depth={fixed_depth}, Horizon={fixed_horizon}ms)'
        )
        ax.set_xlabel('Model')
        ax.set_ylabel('Preprocessing')

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / f'heatmap_{metric}_d{fixed_depth}_h{fixed_horizon}.png',
                dpi=300
            )
            print(f"Saved: heatmap_{metric}_d{fixed_depth}_h{fixed_horizon}.png")

        plt.close()

    def plot_horizon_effect(
        self,
        preprocess: str = 'savgol',
        model: str = 'xgboost',
        depth: int = 10,
        save: bool = True
    ):
        """Plot how accuracy changes with prediction horizon"""
        filtered = self.results_df[
            (self.results_df['preprocess'] == preprocess) &
            (self.results_df['model'] == model) &
            (self.results_df['depth'] == depth)
        ]

        if len(filtered) == 0:
            print(f"No data for {preprocess} + {model}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        horizons = sorted(filtered['horizon_ms'].unique())
        accuracies = [
            filtered[filtered['horizon_ms'] == h]['accuracy'].mean()
            for h in horizons
        ]

        ax.plot(horizons, accuracies, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Prediction Horizon (ms)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy vs Horizon\n({preprocess} + {model}, depth={depth})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / f'horizon_effect_{preprocess}_{model}.png',
                dpi=300
            )
            print(f"Saved: horizon_effect_{preprocess}_{model}.png")

        plt.close()

    def plot_training_time_vs_accuracy(self, save: bool = True):
        """Plot training time vs accuracy scatter"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color by preprocessing method
        for preprocess in self.results_df['preprocess'].unique():
            subset = self.results_df[self.results_df['preprocess'] == preprocess]
            ax.scatter(
                subset['training_time'],
                subset['accuracy'],
                label=preprocess,
                alpha=0.6,
                s=100
            )

        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Time vs Accuracy Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training_time_vs_accuracy.png', dpi=300)
            print("Saved: training_time_vs_accuracy.png")

        plt.close()

    def plot_inference_latency(self, save: bool = True):
        """Plot inference latency comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))

        grouped = self.results_df.groupby('model')['inference_time_ms'].agg(['mean', 'std'])
        grouped = grouped.sort_values('mean')

        x = np.arange(len(grouped))
        ax.barh(x, grouped['mean'], xerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels(grouped.index)
        ax.set_xlabel('Inference Latency (ms per sample)')
        ax.set_title('Model Inference Latency Comparison')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'inference_latency.png', dpi=300)
            print("Saved: inference_latency.png")

        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save: bool = True,
        filename: str = "confusion_matrix.png"
    ):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / filename, dpi=300)
            print(f"Saved: {filename}")

        plt.close()

    def generate_all_plots(self):
        """Generate all standard plots"""
        print("\nGenerating visualizations...")

        # Comparison plots
        self.plot_preprocessing_comparison('accuracy')
        self.plot_preprocessing_comparison('f1_macro')
        self.plot_model_comparison('accuracy')
        self.plot_model_comparison('f1_macro')

        # Heatmap
        if 'depth' in self.results_df.columns and 'horizon_ms' in self.results_df.columns:
            depths = self.results_df['depth'].unique()
            horizons = self.results_df['horizon_ms'].unique()

            if len(depths) > 0 and len(horizons) > 0:
                self.plot_heatmap('accuracy', depths[0], horizons[0])

        # Horizon effect (if multiple horizons)
        if len(self.results_df['horizon_ms'].unique()) > 1:
            preprocess = self.results_df['preprocess'].iloc[0]
            model = self.results_df['model'].iloc[0]
            self.plot_horizon_effect(preprocess, model)

        # Performance trade-offs
        if 'training_time' in self.results_df.columns:
            self.plot_training_time_vs_accuracy()

        if 'inference_time_ms' in self.results_df.columns:
            self.plot_inference_latency()

        print("\n✓ All visualizations generated!")


def plot_preprocessing_effect(
    original: np.ndarray,
    filtered: np.ndarray,
    method: str,
    save_path: Optional[str] = None
):
    """
    Visualize preprocessing effect on time series

    Args:
        original: Original signal
        filtered: Filtered signal
        method: Preprocessing method name
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time series comparison
    t = np.arange(len(original))

    axes[0].plot(t, original, alpha=0.5, label='Original', linewidth=1)
    axes[0].plot(t, filtered, label=f'Filtered ({method})', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mid-price')
    axes[0].set_title(f'Preprocessing Effect: {method}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Noise
    noise = original - filtered
    axes[1].plot(t, noise, color='red', alpha=0.7, linewidth=1)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Noise')
    axes[1].set_title('Removed Noise')
    axes[1].grid(True, alpha=0.3)

    # Compute SNR
    snr = 10 * np.log10(np.var(filtered) / np.var(noise)) if np.var(noise) > 0 else float('inf')
    fig.suptitle(f'SNR: {snr:.2f} dB', fontsize=14, y=1.00)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    """Example usage"""
    print("=== Visualization Module ===\n")

    # Create synthetic results
    np.random.seed(42)

    data = []
    for preprocess in ['raw', 'savgol', 'kalman', 'wavelet', 'ma']:
        for model in ['logistic', 'xgboost', 'catboost']:
            for depth in [10, 20]:
                for horizon in [1000, 5000]:
                    # Simulate results (preprocessing helps)
                    base_acc = 0.50
                    if preprocess != 'raw':
                        base_acc += np.random.uniform(0.01, 0.05)
                    if model == 'xgboost':
                        base_acc += 0.03

                    data.append({
                        'preprocess': preprocess,
                        'model': model,
                        'depth': depth,
                        'horizon_ms': horizon,
                        'accuracy': base_acc + np.random.normal(0, 0.01),
                        'f1_macro': base_acc - 0.02 + np.random.normal(0, 0.01),
                        'mcc': base_acc - 0.1 + np.random.normal(0, 0.02),
                        'training_time': np.random.uniform(1, 30),
                        'inference_time_ms': np.random.uniform(0.1, 5)
                    })

    results_df = pd.DataFrame(data)

    # Create visualizer
    viz = ResultsVisualizer(results_df, output_dir="results/plots")

    # Generate plots
    viz.generate_all_plots()

    # Test preprocessing visualization
    print("\nTesting preprocessing visualization...")
    t = np.linspace(0, 10, 500)
    original = np.sin(2*np.pi*0.5*t) + np.random.normal(0, 0.2, len(t))
    filtered = original - np.random.normal(0, 0.15, len(t))

    plot_preprocessing_effect(
        original,
        filtered,
        'Savitzky-Golay',
        save_path='results/plots/preprocessing_effect_example.png'
    )

    print("\n✓ Visualization module test complete!")


if __name__ == "__main__":
    main()
