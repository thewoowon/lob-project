"""
Main experiment runner for LOB preprocessing study

Runs systematic comparison of:
- Preprocessing methods: raw, savgol, kalman, wavelet, ma
- Models: logistic, xgboost, catboost, cnn, deeplob, cnn_lstm
- LOB depths: 5, 10, 20, 40
- Prediction horizons: 100, 500, 1000, 5000, 10000 ms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from tqdm import tqdm
import yaml
import argparse

from data.download import SyntheticLOBGenerator
from data.preprocess import LOBPreprocessor, MultiLevelPreprocessor
from data.features import LOBFeatureEngineer, create_lob_dataset
from models.baseline import get_baseline_model
from models.deep_models import get_deep_model
from analysis.evaluate import ModelEvaluator, ExperimentTracker
from utils import (
    load_config, setup_logging, ensure_dir, set_random_seed,
    temporal_train_test_split, get_device
)


class ExperimentRunner:
    """Main experiment runner"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize experiment runner"""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['output']['logs_dir'])
        self.tracker = ExperimentTracker()

        # Set random seed
        set_random_seed(self.config['experiments']['random_seed'])

        # Setup device
        self.device = 'cuda' if self.config['experiments']['use_gpu'] else 'cpu'
        if self.device == 'cuda':
            self.device = get_device()

        self.logger.info("Experiment runner initialized")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Device: {self.device}")

    def generate_or_load_data(self) -> pd.DataFrame:
        """Generate synthetic data or load real data"""
        self.logger.info("Generating synthetic LOB data...")

        generator = SyntheticLOBGenerator(
            seed=self.config['experiments']['random_seed']
        )

        df = generator.generate(
            n_snapshots=10000,
            depth=max(self.config['data']['lob_depths']),
            initial_price=50000.0,
            volatility=0.001
        )

        self.logger.info(f"Generated data shape: {df.shape}")
        return df

    def run_single_experiment(
        self,
        df_raw: pd.DataFrame,
        preprocess_method: str,
        model_name: str,
        depth: int,
        horizon_ms: int
    ) -> dict:
        """
        Run a single experiment configuration

        Returns:
            Dictionary with experiment results
        """
        self.logger.info(
            f"Running: {preprocess_method} | {model_name} | "
            f"depth={depth} | horizon={horizon_ms}ms"
        )

        try:
            # Step 1: Preprocessing
            if preprocess_method != 'raw':
                preprocessor = MultiLevelPreprocessor(method=preprocess_method)
                df_processed = preprocessor.preprocess_lob_dataframe(df_raw, depth=depth)
            else:
                df_processed = df_raw.copy()

            # Step 2: Feature extraction
            engineer = LOBFeatureEngineer(depth=depth)
            features = engineer.extract_all_features(df_processed)

            # Step 3: Create labels
            labels = engineer.create_labels(
                df_processed,
                horizon_ms=horizon_ms,
                threshold=self.config['prediction']['threshold_ticks'],
                task=self.config['prediction']['task']
            )

            # Align features and labels
            steps = max(1, horizon_ms // 100)
            features = features.iloc[:-steps]

            # Remove NaN
            mask = ~(features.isna().any(axis=1) | pd.isna(labels))
            X = features[mask].values
            y = labels[mask]

            if len(X) == 0:
                raise ValueError("No valid samples after preprocessing")

            # Step 4: Train/val/test split
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
                X, y,
                train_ratio=self.config['prediction']['train_ratio'],
                val_ratio=self.config['prediction']['val_ratio'],
                test_ratio=self.config['prediction']['test_ratio']
            )

            self.logger.info(
                f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

            # Step 5: Train model
            num_classes = len(np.unique(y))

            if model_name in ['logistic', 'xgboost', 'catboost', 'lightgbm']:
                # Baseline model
                model = get_baseline_model(
                    model_name,
                    **self.config['models'].get(model_name, {})
                )
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)
                training_time = model.training_time
                inference_time = model.inference_time

            else:
                # Deep learning model
                trainer = get_deep_model(
                    model_name,
                    input_size=X_train.shape[1],
                    num_classes=num_classes,
                    device=self.device,
                    **self.config['models'].get(model_name, {})
                )

                # Train
                epochs = self.config['models'][model_name].get('epochs', 50)
                trainer.fit(X_train, y_train, X_val, y_val, epochs=epochs, verbose=False)

                # Predict
                y_pred = trainer.predict(X_test)
                training_time = trainer.training_time
                inference_time = trainer.inference_time

            # Step 6: Evaluate
            evaluator = ModelEvaluator(num_classes=num_classes)
            metrics = evaluator.evaluate(y_test, y_pred)

            # Log to tracker
            self.tracker.log_experiment(
                preprocess_method=preprocess_method,
                model_name=model_name,
                depth=depth,
                horizon_ms=horizon_ms,
                metrics=metrics,
                training_time=training_time,
                inference_time=inference_time
            )

            self.logger.info(
                f"Results: acc={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1_macro']:.4f}, "
                f"mcc={metrics['mcc']:.4f}"
            )

            return {
                'success': True,
                'metrics': metrics,
                'training_time': training_time,
                'inference_time': inference_time
            }

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def run_all_experiments(
        self,
        preprocess_methods: list = None,
        models: list = None,
        depths: list = None,
        horizons: list = None
    ):
        """
        Run all experiment combinations

        Args:
            preprocess_methods: List of preprocessing methods (or None for all)
            models: List of models (or None for all)
            depths: List of depths (or None for all from config)
            horizons: List of horizons in ms (or None for all from config)
        """
        # Use config defaults if not specified
        if preprocess_methods is None:
            preprocess_methods = self.config['preprocessing']['methods']
        if models is None:
            models = ['logistic', 'xgboost', 'catboost']  # Default to fast models
        if depths is None:
            depths = self.config['data']['lob_depths']
        if horizons is None:
            horizons = self.config['prediction']['horizons_ms']

        # Generate data once
        self.logger.info("=" * 60)
        self.logger.info("STARTING EXPERIMENT SUITE")
        self.logger.info("=" * 60)

        df_raw = self.generate_or_load_data()

        # Create all combinations
        experiments = list(itertools.product(
            preprocess_methods,
            models,
            depths,
            horizons
        ))

        self.logger.info(f"Total experiments to run: {len(experiments)}")

        # Run experiments
        with tqdm(total=len(experiments), desc="Running experiments") as pbar:
            for preprocess, model, depth, horizon in experiments:
                self.run_single_experiment(
                    df_raw=df_raw,
                    preprocess_method=preprocess,
                    model_name=model,
                    depth=depth,
                    horizon_ms=horizon
                )
                pbar.update(1)

        # Save results
        results_dir = ensure_dir(self.config['output']['results_dir'])
        results_path = results_dir / "experiment_results.csv"
        self.tracker.save_results(results_path)

        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUITE COMPLETE")
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info("=" * 60)

    def run_quick_test(self):
        """Run a quick test with minimal configurations"""
        self.logger.info("Running quick test...")

        self.run_all_experiments(
            preprocess_methods=['raw', 'savgol'],
            models=['logistic', 'xgboost'],
            depths=[10],
            horizons=[1000]
        )

    def analyze_results(self):
        """Analyze and print summary of results"""
        df = self.tracker.get_results_dataframe()

        if len(df) == 0:
            print("No results to analyze")
            return

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        # Overall best
        best = self.tracker.get_best_config('accuracy')
        print(f"\nBest configuration (by accuracy):")
        print(f"  Preprocessing: {best['preprocess']}")
        print(f"  Model: {best['model']}")
        print(f"  Depth: {best['depth']}")
        print(f"  Horizon: {best['horizon_ms']}ms")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  F1 (macro): {best['f1_macro']:.4f}")

        # Best per preprocessing method
        print("\n\nBest accuracy per preprocessing method:")
        for method in df['preprocess'].unique():
            subset = df[df['preprocess'] == method]
            best_acc = subset['accuracy'].max()
            best_row = subset[subset['accuracy'] == best_acc].iloc[0]
            print(f"  {method:10s}: {best_acc:.4f} ({best_row['model']})")

        # Best per model
        print("\n\nBest accuracy per model:")
        for model in df['model'].unique():
            subset = df[df['model'] == model]
            best_acc = subset['accuracy'].max()
            best_row = subset[subset['accuracy'] == best_acc].iloc[0]
            print(f"  {model:10s}: {best_acc:.4f} ({best_row['preprocess']})")

        # Average by preprocessing
        print("\n\nAverage metrics by preprocessing method:")
        print(df.groupby('preprocess')[['accuracy', 'f1_macro', 'mcc']].mean().round(4))

        # Average by model
        print("\n\nAverage metrics by model:")
        print(df.groupby('model')[['accuracy', 'f1_macro', 'training_time']].mean().round(4))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run LOB preprocessing experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with minimal configs')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing results')

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(config_path=args.config)

    if args.analyze:
        # Load existing results
        results_path = Path(runner.config['output']['results_dir']) / "experiment_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            # Reconstruct tracker
            for _, row in df.iterrows():
                metrics = row.to_dict()
                runner.tracker.log_experiment(
                    preprocess_method=row['preprocess'],
                    model_name=row['model'],
                    depth=row['depth'],
                    horizon_ms=row['horizon_ms'],
                    metrics=metrics,
                    training_time=row.get('training_time', 0),
                    inference_time=row.get('inference_time_ms', 0) / 1000
                )
        runner.analyze_results()

    elif args.quick:
        runner.run_quick_test()
        runner.analyze_results()

    else:
        runner.run_all_experiments()
        runner.analyze_results()


if __name__ == "__main__":
    main()
