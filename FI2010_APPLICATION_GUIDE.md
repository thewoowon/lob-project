# 📋 FI-2010 데이터셋 신청 가이드

## 🎯 FI-2010이란?

**공식 명칭**: Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data

### 데이터셋 상세 정보

```
📊 데이터 규모:
- 5개 핀란드 주식 (NASDAQ Nordic)
- 10 거래일 (2010년 6월 1일 ~ 6월 14일)
- 약 4,000,000개 time series samples
- 10 levels LOB data (bid/ask 각 10개)

🎨 Normalization:
- 3가지 방법 제공: Z-score, Min-Max, Decimal-precision
- 이미 전처리된 버전 포함

⏱️ Prediction Horizons:
- 5가지: 10, 20, 30, 50, 100 ticks
- 우리 실험(100ms)과 비교 가능

📜 License:
- Creative Commons Attribution 4.0
- 학술 목적 무료 사용 가능
```

### 우리 연구에 완벽한 이유 ✅

```
✅ Real LOB L2 orderbook data (10 levels)
✅ 우리 실험 구조와 동일 (depth, horizon)
✅ 학술적으로 검증된 benchmark
✅ 다른 논문들도 사용 (비교 가능)
✅ 무료 (학술 목적)
✅ 즉시 다운로드 가능 (승인 불필요!)
```

---

## 📥 다운로드 방법

### Option 1: Fairdata 공식 사이트 (추천)

**링크**: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

**절차**:
1. 위 링크 클릭
2. JavaScript 활성화 필요
3. "Download" 버튼 클릭
4. 데이터 다운로드

⚠️ **참고**: JavaScript 필요하므로 일반 브라우저에서 접속

### Option 2: GitHub 구현체 사용

여러 GitHub 저장소에서 FI-2010 데이터 로딩 코드 제공:

```bash
# Example: lob-deep-learning repository
git clone https://github.com/Jeonghwan-Cheon/lob-deep-learning
cd lob-deep-learning
# README 참고하여 데이터 다운로드
```

### Option 3: ArXiv Paper에서 링크 확인

**Paper**: https://arxiv.org/abs/1705.03233

논문에서 데이터셋 다운로드 링크 확인 가능

---

## 🚀 신청 절차 (Step-by-Step)

### Step 1: 브라우저에서 접속
```
URL: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

Browser: Chrome, Firefox, Safari 등
⚠️ JavaScript 활성화 필수
```

### Step 2: 데이터셋 정보 확인
```
페이지에서 확인할 것:
- Dataset description
- File formats (likely CSV or similar)
- Total size
- License agreement
```

### Step 3: 다운로드
```
1. "Download" 또는 "Access" 버튼 클릭
2. 필요시 email 등록 (간단한 form)
3. License agreement 동의
4. 파일 다운로드
```

### Step 4: 데이터 검증
```bash
# 다운로드 후 확인
cd /Users/aepeul/lob-project/lob_preprocessing/data/raw
mkdir fi2010
cd fi2010

# 압축 해제 (zip/tar.gz 등)
unzip FI2010_Dataset.zip  # or tar -xzf FI2010_Dataset.tar.gz

# 파일 확인
ls -lh
head *.csv  # CSV인 경우
```

---

## 📊 예상 데이터 구조

### FI-2010 Format (예상)
```csv
timestamp, bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ..., bid_price_10, bid_vol_10, ask_price_10, ask_vol_10, mid_price, label_10, label_20, label_30, label_50, label_100
```

**Features (40 columns)**:
- 10 levels × 2 sides × 2 values (price, volume) = 40 features
- Plus: labels for different horizons

### 우리 실험 형식으로 변환
```python
# FI-2010 → Our format
def load_fi2010(file_path):
    df = pd.read_csv(file_path)

    # Extract LOB features
    lob_features = df.iloc[:, :40]  # First 40 columns
    labels = df['label_100']  # 100-tick horizon

    return lob_features, labels
```

---

## 🔧 우리 코드에 통합

### Step 1: Data Loader 추가

```python
# data/download.py에 추가

class FI2010Loader:
    """FI-2010 데이터셋 로더"""

    def __init__(self, data_dir='data/raw/fi2010'):
        self.data_dir = Path(data_dir)

    def load_stock(self, stock_id, day):
        """
        Load one stock for one day

        Args:
            stock_id: 1-5
            day: 1-10
        """
        file_path = self.data_dir / f'stock_{stock_id}_day_{day}.csv'
        df = pd.read_csv(file_path)

        # Extract features and labels
        features = df.iloc[:, :40].values  # LOB features
        labels = df['label_100'].values  # 100-tick labels

        return features, labels

    def load_all(self):
        """Load all stocks and days"""
        all_features = []
        all_labels = []

        for stock_id in range(1, 6):
            for day in range(1, 11):
                features, labels = self.load_stock(stock_id, day)
                all_features.append(features)
                all_labels.append(labels)

        return np.vstack(all_features), np.concatenate(all_labels)
```

### Step 2: 실험 스크립트 수정

```python
# experiments/run_fi2010_validation.py (새 파일)

from data.download import FI2010Loader
from data.preprocess import LOBPreprocessor
from models.baseline import XGBoostModel, CatBoostModel
import pandas as pd

def run_fi2010_validation():
    """FI-2010으로 핵심 config 검증"""

    print("🔬 FI-2010 VALIDATION")
    print("="*60)

    # Load FI-2010 data
    loader = FI2010Loader()
    X, y = loader.load_all()

    print(f"✅ Loaded FI-2010: {X.shape[0]} samples")

    # Train/test split (temporal)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Key configs
    configs = [
        ('raw', 'xgboost'),
        ('wavelet', 'xgboost'),
        ('kalman', 'xgboost'),
        ('raw', 'catboost'),
        ('wavelet', 'catboost'),
    ]

    results = []

    for preprocess, model_name in configs:
        print(f"\n▶ {preprocess.upper()} + {model_name.upper()}")

        # Preprocess
        if preprocess != 'raw':
            preprocessor = LOBPreprocessor(method=preprocess)
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)
        else:
            X_train_proc = X_train
            X_test_proc = X_test

        # Train
        if model_name == 'xgboost':
            model = XGBoostModel()
        else:
            model = CatBoostModel()

        model.fit(X_train_proc, y_train)

        # Evaluate
        y_pred = model.predict(X_test_proc)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1-Macro: {f1:.4f}")

        results.append({
            'preprocess': preprocess,
            'model': model_name,
            'accuracy': acc,
            'f1_macro': f1
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/fi2010_validation.csv', index=False)

    print(f"\n✅ Results saved to results/fi2010_validation.csv")

    return results_df

if __name__ == '__main__':
    run_fi2010_validation()
```

---

## 📈 예상 결과

### Realistic Expectations

```
Synthetic (우리 결과):
- Raw + XGBoost: 53.55%
- Wavelet + XGBoost: 85.15%
- Improvement: +31.6%

FI-2010 (예상):
- Raw + XGBoost: 45-55% (benchmark 논문 결과)
- Wavelet + XGBoost: 55-65% (예상)
- Improvement: +10-15%

→ 여전히 유의미한 개선!
→ 전처리 효과 검증됨
```

### 논문에 쓸 내용
```
"We validate our findings on the FI-2010 benchmark dataset.
 While the absolute accuracy is lower than synthetic data
 (as expected), the relative improvement from preprocessing
 remains consistent (+10-15%), confirming our hypothesis
 that preprocessing is critical for LOB prediction."
```

---

## 📝 논문 업데이트 계획

### Current (Synthetic only)
```
Results:
- Wavelet + XGBoost: 85.30%
- Raw baseline: 52.74%
- Improvement: +32.56%
```

### Updated (Synthetic + FI-2010)
```
Results:

1. Synthetic Data (Controlled Environment)
   - Wavelet + XGBoost: 85.30%
   - Raw baseline: 52.74%
   - Improvement: +32.56%

2. FI-2010 Validation (Real Data)
   - Wavelet + XGBoost: 58.3% (example)
   - Raw baseline: 48.7%
   - Improvement: +9.6% (+19.7% relative)

3. Analysis
   - Preprocessing effect consistent across datasets
   - Absolute accuracy differs (synthetic vs real)
   - Relative improvement validates our hypothesis
```

---

## ⏱️ Timeline

### TODAY (30분)
```
1. 브라우저에서 FI-2010 사이트 접속
2. 데이터 다운로드
3. 파일 구조 확인
```

### TOMORROW (2시간)
```
4. FI2010Loader 구현
5. 데이터 로딩 테스트
6. 형식 확인 및 변환
```

### 2-3 DAYS (4시간)
```
7. 핵심 5개 config 실행
8. 결과 분석
9. Synthetic vs FI-2010 비교
```

### WEEK 1 END
```
10. 논문 Results 섹션 업데이트
11. Discussion 작성 (두 데이터셋 비교)
12. 교수 미팅 (결과 보고)
```

---

## 📚 참고 자료

### Paper
- **ArXiv**: https://arxiv.org/abs/1705.03233
- **Journal**: Ntakaris et al. (2018), Journal of Forecasting

### Dataset
- **Official**: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- **License**: Creative Commons Attribution 4.0

### Implementations
- **GitHub**: https://github.com/Jeonghwan-Cheon/lob-deep-learning
- **DeepAI**: https://deepai.org/publication/benchmark-dataset-for-mid-price-prediction-of-limit-order-book-data

---

## ✅ Action Items

### 🚨 RIGHT NOW (10분)
```
[ ] 브라우저 열기
[ ] https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 접속
[ ] "Download" 버튼 찾기
[ ] 파일 다운로드 시작
```

### 📅 TODAY
```
[ ] 다운로드 완료
[ ] 압축 해제
[ ] data/raw/fi2010/ 폴더에 저장
[ ] 파일 구조 확인 (head 명령어)
[ ] 내게 결과 보고 (파일 구조 공유)
```

### 📅 THIS WEEK
```
[ ] FI2010Loader 구현
[ ] 핵심 실험 5개 실행
[ ] 결과 비교 (Synthetic vs FI-2010)
[ ] 교수 미팅 준비
```

---

## 💡 Pro Tips

### Tip 1: 데이터 크기
```
FI-2010은 4M samples → 클 수 있음
메모리 부족 시:
- 한 번에 1 stock만 로드
- 또는 downsampling
```

### Tip 2: 형식 불일치
```
FI-2010 형식이 우리와 다를 수 있음
→ 천천히 변환 로직 작성
→ 작은 샘플로 먼저 테스트
```

### Tip 3: Label 차이
```
FI-2010: tick-based labels (10, 20, ... ticks)
우리: time-based (100ms, 500ms, ...)

→ 100-tick label 사용 (가장 가까움)
→ 또는 time conversion
```

---

## 🎉 화이팅!

**FI-2010 확보하면:**
- ✅ Real LOB data 검증 완료
- ✅ 교수 설득 가능
- ✅ 논문 훨씬 강력해짐
- ✅ 졸업 확정적

**지금 바로 다운로드 시작하자! 🚀**

---

**Sources:**
- [Benchmark Dataset Paper (ArXiv)](https://arxiv.org/abs/1705.03233)
- [FI-2010 Official Dataset (Fairdata)](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
- [Journal Publication (Wiley)](https://onlinelibrary.wiley.com/doi/full/10.1002/for.2543)
- [Implementation Examples (GitHub)](https://github.com/Jeonghwan-Cheon/lob-deep-learning)
- [ResearchGate Discussion](https://www.researchgate.net/publication/316821343_Benchmark_Dataset_for_Mid-Price_Prediction_of_Limit_Order_Book_data)
