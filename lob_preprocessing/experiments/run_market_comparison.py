"""
크립토 vs 한국 주식 시장 비교 실험

Research Questions:
1. 전처리 효과가 시장별로 다른가? (Crypto vs Korean)
2. Liquid (BTC, 삼성전자) vs Illiquid (KOSDAQ 소형주) 차이는?
3. 24/7 market (Crypto) vs Market hours (Korean) 영향은?
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml

from data.unified_loader import UnifiedLOBLoader
from data.preprocess import LOBPreprocessor
from data.features import LOBFeatureEngineer, create_lob_dataset
from models.baseline import get_baseline_model
from analysis.evaluate import ModelEvaluator, ExperimentTracker
from utils import set_random_seed, temporal_train_test_split


class MarketComparisonExperiment:
    """크립토 vs 한국 주식 비교 실험"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        set_random_seed(42)

        self.loader = UnifiedLOBLoader()
        self.tracker = ExperimentTracker()

    def run_single_market_experiment(
        self,
        df: pd.DataFrame,
        market_name: str,
        preprocess_method: str,
        model_name: str,
        horizon_ms: int
    ) -> Dict:
        """단일 시장 실험"""
        print(f"\nRunning: {market_name} | {preprocess_method} | {model_name} | {horizon_ms}ms")

        try:
            # 전처리
            if preprocess_method != 'raw':
                preprocessor = LOBPreprocessor(method=preprocess_method)
                df_processed = df.copy()
                df_processed['mid_price'] = preprocessor.fit_transform(df['mid_price'].values)

                # SNR 계산
                snr = preprocessor.compute_snr(
                    df['mid_price'].values,
                    df_processed['mid_price'].values
                )
            else:
                df_processed = df.copy()
                snr = 0.0

            # Feature extraction
            engineer = LOBFeatureEngineer(depth=5)
            features = engineer.extract_all_features(df_processed)
            labels = engineer.create_labels(
                df_processed,
                horizon_ms=horizon_ms,
                threshold=0.5,
                task='ternary'
            )

            # Align
            steps = max(1, horizon_ms // 100)
            features = features.iloc[:-steps]

            mask = ~(features.isna().any(axis=1) | pd.isna(labels))
            X = features[mask].values
            y = labels[mask]

            # Split
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
                X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
            )

            # Train model
            model = get_baseline_model(model_name)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            evaluator = ModelEvaluator(num_classes=3)
            metrics = evaluator.evaluate(y_test, y_pred)
            metrics['snr'] = snr

            # Log
            result = {
                'market': market_name,
                'preprocess': preprocess_method,
                'model': model_name,
                'horizon_ms': horizon_ms,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'mcc': metrics['mcc'],
                'snr': snr,
                'training_time': model.training_time,
                'inference_time_ms': model.inference_time * 1000
            }

            print(f"  ✓ Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}, SNR: {snr:.2f} dB")

            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def run_market_comparison(
        self,
        crypto_symbol: str = 'BTCUSDT',
        korean_code: str = '005930',
        start_date: str = '2024-01-01',
        end_date: str = '2024-01-31',
        preprocess_methods: List[str] = None,
        models: List[str] = None,
        horizons: List[int] = None
    ):
        """시장 비교 실험 실행"""

        if preprocess_methods is None:
            preprocess_methods = ['raw', 'savgol', 'kalman']
        if models is None:
            models = ['logistic', 'xgboost']
        if horizons is None:
            horizons = [1000, 5000]

        print("=" * 60)
        print("MARKET COMPARISON EXPERIMENT")
        print("=" * 60)
        print(f"Crypto: {crypto_symbol}")
        print(f"Korean: {korean_code}")
        print(f"Period: {start_date} ~ {end_date}")
        print(f"Preprocessing: {preprocess_methods}")
        print(f"Models: {models}")
        print(f"Horizons: {horizons} ms")
        print("=" * 60)

        # 데이터 로드
        try:
            data = self.loader.compare_markets(
                crypto_symbol=crypto_symbol,
                korean_code=korean_code,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            print(f"Data loading failed: {e}")
            print("\nUsing synthetic data for demonstration...")

            # Synthetic fallback
            from data.download import SyntheticLOBGenerator
            generator = SyntheticLOBGenerator(seed=42)

            crypto_df = generator.generate(n_snapshots=5000, depth=5)
            crypto_df['market'] = 'crypto'
            crypto_df['code'] = crypto_symbol

            korean_df = generator.generate(n_snapshots=3000, depth=5)
            korean_df['market'] = 'kospi'
            korean_df['code'] = korean_code

            data = {'crypto': crypto_df, 'korean': korean_df}

        # 실험 실행
        results = []

        for market_name, df in data.items():
            print(f"\n\n{'='*60}")
            print(f"Market: {market_name.upper()} ({len(df)} samples)")
            print('='*60)

            for preprocess in preprocess_methods:
                for model in models:
                    for horizon in horizons:
                        result = self.run_single_market_experiment(
                            df=df,
                            market_name=market_name,
                            preprocess_method=preprocess,
                            model_name=model,
                            horizon_ms=horizon
                        )

                        if result:
                            results.append(result)

        # 결과 저장
        results_df = pd.DataFrame(results)
        output_dir = Path('results/market_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / 'comparison_results.csv'
        results_df.to_csv(results_path, index=False)

        print("\n\n" + "=" * 60)
        print("RESULTS SAVED")
        print("=" * 60)
        print(f"File: {results_path}")

        # 요약 분석
        self.analyze_market_comparison(results_df)

    def analyze_market_comparison(self, df: pd.DataFrame):
        """시장 비교 분석"""
        print("\n\n" + "=" * 60)
        print("MARKET COMPARISON ANALYSIS")
        print("=" * 60)

        # 1. 시장별 평균 성능
        print("\n1. Average Performance by Market:")
        market_perf = df.groupby('market')[['accuracy', 'f1_macro', 'snr']].mean()
        print(market_perf.round(4))

        # 2. 전처리 효과 (시장별)
        print("\n2. Preprocessing Effect by Market:")
        prep_effect = df.groupby(['market', 'preprocess'])['accuracy'].mean().unstack()
        print(prep_effect.round(4))

        # 시장별 전처리 개선율
        print("\n3. Preprocessing Improvement (vs raw):")
        for market in df['market'].unique():
            market_df = df[df['market'] == market]
            raw_acc = market_df[market_df['preprocess'] == 'raw']['accuracy'].mean()

            print(f"\n  {market.upper()}:")
            for prep in ['savgol', 'kalman', 'wavelet']:
                prep_df = market_df[market_df['preprocess'] == prep]
                if len(prep_df) > 0:
                    prep_acc = prep_df['accuracy'].mean()
                    improvement = (prep_acc - raw_acc) / raw_acc * 100
                    print(f"    {prep:10s}: {improvement:+.2f}%")

        # 4. 모델별 성능 (시장별)
        print("\n4. Model Performance by Market:")
        model_perf = df.groupby(['market', 'model'])['accuracy'].mean().unstack()
        print(model_perf.round(4))

        # 5. SNR 분석
        print("\n5. SNR Analysis:")
        snr_analysis = df[df['preprocess'] != 'raw'].groupby(['market', 'preprocess'])['snr'].mean().unstack()
        print(snr_analysis.round(2))

        # 6. Best configurations
        print("\n6. Best Configurations:")
        for market in df['market'].unique():
            market_df = df[df['market'] == market]
            best = market_df.nlargest(1, 'accuracy').iloc[0]

            print(f"\n  {market.upper()}:")
            print(f"    Preprocessing: {best['preprocess']}")
            print(f"    Model: {best['model']}")
            print(f"    Accuracy: {best['accuracy']:.4f}")
            print(f"    F1: {best['f1_macro']:.4f}")


def main():
    """실행"""
    import argparse

    parser = argparse.ArgumentParser(description='Market Comparison Experiment')
    parser.add_argument('--crypto', type=str, default='BTCUSDT',
                       help='Crypto symbol')
    parser.add_argument('--korean', type=str, default='005930',
                       help='Korean stock code')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-31',
                       help='End date')

    args = parser.parse_args()

    # 실험 실행
    experiment = MarketComparisonExperiment()
    experiment.run_market_comparison(
        crypto_symbol=args.crypto,
        korean_code=args.korean,
        start_date=args.start,
        end_date=args.end,
        preprocess_methods=['raw', 'savgol', 'kalman'],
        models=['logistic', 'xgboost'],
        horizons=[1000, 5000]
    )


if __name__ == "__main__":
    main()
