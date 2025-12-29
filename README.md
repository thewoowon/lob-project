# **틈새 #2 완전 가이드: LOB 데이터 전처리 vs 모델 복잡도**

---

## **I. 연구 배경 및 동기**

### **핵심 발견**
2025년 6월 최신 연구: cryptocurrency LOB에서 데이터 전처리 후 XGBoost 같은 단순 모델이 복잡한 neural networks를 1-2% 상회. Binary/ternary 예측 정확도 0.42-0.71 (100ms-1000ms 예측)

핵심 주장: "Better inputs matter more than stacking another hidden layer" - 모델 복잡도보다 input 품질이 중요

### **연구 갭 (Research Gap)**
1. **체계적 비교 부족**: 어떤 전처리가 언제 왜 효과적인지 명확하지 않음
2. **자산별 차이**: Liquid vs illiquid, crypto vs equity 비교 부족
3. **이론적 설명 부족**: 왜 전처리가 도움이 되는지 메커니즘 불명확
4. **한국 시장 연구 전무**: KOSPI/KOSDAQ LOB 데이터 연구 없음
5. **실무 가이드라인 부재**: Practitioners를 위한 체계적 지침 없음

---

## **II. 연구 질문 (Research Questions)**

### **Main RQ:**
**"어떤 preprocessing 방법이 어떤 조건에서 LOB mid-price 예측 성능을 향상시키는가?"**

### **Sub-RQs:**
1. **RQ1**: Savitzky-Golay vs Kalman vs Wavelet - 어느 것이 언제 우수한가?
2. **RQ2**: Liquid stocks vs illiquid stocks - 전처리 효과가 다른가?
3. **RQ3**: 단순 모델 vs 복잡 모델 - 전처리가 어디에 더 도움이 되는가?
4. **RQ4**: 예측 horizon (100ms vs 1s vs 10s) - 전처리 효과가 달라지는가?
5. **RQ5**: **WHY** - Signal-to-noise ratio 개선이 메커니즘인가?

---

## **III. 방법론 (Methodology)**

### **A. 데이터**

**1. Cryptocurrency (Primary)**
- **Source**: Bybit public historical data
- **Assets**: BTC/USDT, ETH/USDT
- **Period**: 1-3개월
- **Frequency**: 100ms snapshots
- **Depth**: 5, 10, 20, 40 levels
- **장점**: 무료, 24/7 거래, high-frequency

**2. 한국 주식 (Secondary, if possible)**
- **Assets**: 
  - Liquid: 삼성전자 (005930)
  - Illiquid: KOSDAQ 중소형주 1종
- **Source**: 
  - KOSCOM (유료)
  - 또는 크래프톤/증권사 API
- **Challenge**: 데이터 접근성

**3. 벤치마크 데이터**
- **NASDAQ**: INTC (LOBSTER) - 기존 연구와 비교용

---

### **B. 전처리 방법 (Preprocessing Methods)**

**1. Savitzky-Golay Filter**
```python
from scipy.signal import savgol_filter
# Window size: 5, 11, 21
# Polynomial order: 2, 3
filtered = savgol_filter(mid_price, window_length=11, polyorder=2)
```
- **특징**: Local polynomial regression
- **장점**: Edge-preserving, 빠름
- **단점**: Window size 선택 민감

**2. Kalman Filter**
```python
from pykalman import KalmanFilter
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
filtered_state_means, _ = kf.filter(mid_price)
```
- **특징**: Recursive Bayesian estimation
- **장점**: Real-time 적용 가능, 이론적 기반
- **단점**: Parameter tuning 필요

**3. Wavelet Denoising**
```python
import pywt
coeffs = pywt.wavedec(mid_price, 'db4', level=3)
# Soft thresholding
filtered = pywt.waverec(coeffs_thresh, 'db4')
```
- **특징**: Multi-resolution analysis
- **장점**: Frequency-domain filtering
- **단점**: Computationally expensive

**4. Moving Average (Baseline)**
```python
# Simple MA, Exponential MA
ma = mid_price.rolling(window=10).mean()
ema = mid_price.ewm(span=10).mean()
```

**5. Raw Data (Control)**
- No preprocessing

---

### **C. Feature Engineering**

**기본 Features (Existing literature):**
1. **Price features**:
   - Mid-price: $(P_{ask}^1 + P_{bid}^1) / 2$
   - Spread: $P_{ask}^1 - P_{bid}^1$
   - Microprice: Volume-weighted

2. **Volume features**:
   - Order imbalance: $\frac{V_{bid} - V_{ask}}{V_{bid} + V_{ask}}$
   - Total volume at each level

3. **Order Flow Imbalance (OFI)**:
   $OFI = \sum_{levels} \Delta(volume \times price)$

4. **Time features**:
   - Time since last trade
   - Volatility (rolling std)

**Your contribution: 전처리 적용 후 features**

---

### **D. Models**

**Simple Models:**
1. **Logistic Regression** (Baseline)
2. **XGBoost**
   ```python
   xgb.XGBClassifier(
       max_depth=3,
       learning_rate=0.1,
       n_estimators=100
   )
   ```
3. **CatBoost**

**Deep Learning Models:**
4. **Simple CNN**
   ```python
   Conv1D(64, 3) -> ReLU -> BatchNorm
   Conv1D(64, 3) -> ReLU -> BatchNorm
   GlobalMaxPooling -> Dense(32) -> Output
   ```

5. **DeepLOB** (Benchmark)
   Zhang et al. 2019 architecture - 기존 연구 표준

6. **Conv1D + LSTM**

---

### **E. 실험 설계**

**Prediction Tasks:**
- **Binary**: Up/Down (mid-price change > 0)
- **Ternary**: Up/Flat/Down (threshold θ = 1 tick)

**Prediction Horizons:**
- T = 100ms, 500ms, 1s, 5s, 10s

**Train/Val/Test Split:**
- Train: 60%
- Validation: 20%
- Test: 20%
- **중요**: Temporal split (no look-ahead bias)

**Ablation Study:**
| Experiment | Preprocessing | Model | LOB Depth |
|------------|--------------|-------|-----------|
| Exp1 | None (raw) | All models | 5/10/20/40 |
| Exp2 | Savitzky-Golay | All models | 5/10/20/40 |
| Exp3 | Kalman | All models | 5/10/20/40 |
| Exp4 | Wavelet | All models | 5/10/20/40 |
| Exp5 | Moving Average | All models | 5/10/20/40 |

**Total experiments**: 5 preprocessing × 6 models × 4 depths × 5 horizons = **600 configurations**

---

### **F. 평가 지표**

**Prediction Performance:**
1. **Accuracy**
2. **F1-Score** (class imbalance 고려)
3. **Matthews Correlation Coefficient (MCC)**
4. **Confusion Matrix**

**Signal Quality:**
5. **Signal-to-Noise Ratio (SNR)**:
   $$SNR = 10 \log_{10}\left(\frac{\sigma_{signal}^2}{\sigma_{noise}^2}\right)$$

6. **Autocorrelation** (before/after preprocessing)

**Computational Efficiency:**
7. **Training Time**
8. **Inference Latency** (ms per prediction)
9. **Memory Usage**

**Trading Simulation (Optional, if time permits):**
10. **Sharpe Ratio**
11. **Maximum Drawdown**
12. **Total Return**

---

## **IV. 예상 결과 (Expected Findings)**

### **Hypothesis 1: 전처리 효과는 자산 유동성에 따라 다르다**
- **Illiquid assets**: 전처리 효과 큼 (noise 많음)
- **Liquid assets**: 전처리 효과 작음 (signal 이미 강함)

### **Hypothesis 2: 단순 모델이 전처리로 더 많이 개선된다**
- XGBoost + preprocessing > DeepLOB (raw)
- Deep models는 내부적으로 denoising 학습

### **Hypothesis 3: Savitzky-Golay가 가장 practical**
- Latency vs accuracy trade-off에서 최적
- Kalman은 이론적으로 우수하지만 느림

### **Hypothesis 4: Short horizon에서 전처리 효과 큼**
- T=100ms: 전처리 필수
- T=10s: 전처리 효과 감소 (장기 트렌드가 noise 압도)

---

## **V. 기여도 (Contributions)**

### **학술적 기여:**
1. ✅ **첫 체계적 LOB 전처리 비교 연구**
2. ✅ **이론적 메커니즘 설명** (SNR 분석)
3. ✅ **자산별/horizon별 차이 규명**
4. ✅ **한국 시장 첫 LOB microstructure 연구**

### **실무적 기여:**
5. ✅ **Practitioner guideline**: "언제 어떤 전처리를 쓸 것인가"
6. ✅ **오픈소스 코드** (재현 가능성)
7. ✅ **Latency-aware recommendations**

### **기술적 기여:**
8. ✅ **Efficient pipeline** (LOBFrame 확장)
9. ✅ **Comparative framework** for future research

---

## **VI. 출판 전략 (Publication Strategy)**

### **Target 1: 국내 학회 (확실)**
- **한국금융공학회 학술대회** (연 2회)
- **한국경영과학회 추계학술대회**
- **한국데이터정보과학회**
- **장점**: 빠른 피드백, 한국어 발표 가능, 네트워킹

### **Target 2: International Conference**
- **ICAIF (ACM Int'l Conf on AI in Finance)** - Top tier
- **KDD Workshop on Financial Data Science**
- **NeurIPS Workshop on ML in Finance**

### **Target 3: Journal (SCI/SSCI)**
- **Tier 1 (도전)**:
  - Journal of Computational Finance
  - Quantitative Finance
  
- **Tier 2 (현실적)**:
  - Expert Systems with Applications (SCI)
  - Applied Soft Computing (SCI)
  - Finance Research Letters (SSCI)
  
- **Tier 3 (안전)**:
  - 한국경영과학회지 (KCI)
  - 재무연구 (KCI)

---

## **VII. 타임라인 (12주 계획)**

### **Week 1-2: 준비 단계**
- [ ] Literature review 완료
- [ ] 데이터 다운로드 (Bybit)
- [ ] 환경 설정 (Python, GPU)
- [ ] LOBFrame 설치 및 테스트

### **Week 3-4: 데이터 전처리**
- [ ] Raw LOB 데이터 파싱
- [ ] 5가지 전처리 구현
- [ ] Feature extraction
- [ ] Train/val/test split

### **Week 5-6: 모델 구현**
- [ ] 6개 모델 구현
- [ ] Hyperparameter tuning (validation set)
- [ ] Baseline 결과 확보

### **Week 7-8: 실험 실행**
- [ ] 600 configurations 실험
- [ ] 결과 로깅 (MLflow/WandB)
- [ ] Intermediate 분석

### **Week 9-10: 분석 및 해석**
- [ ] SNR 분석
- [ ] Statistical significance tests
- [ ] Ablation study 결과 정리
- [ ] Visualization (heatmaps, confusion matrices)

### **Week 11: 논문 작성**
- [ ] Introduction
- [ ] Methodology
- [ ] Results
- [ ] Discussion
- [ ] Conclusion

### **Week 12: 최종 점검**
- [ ] 교수 피드백 반영
- [ ] 코드 정리 (GitHub)
- [ ] Supplementary materials
- [ ] 최종 제출

---

## **VIII. 구현 상세 (Implementation Details)**

### **A. 코드 구조**
```
lob_preprocessing/
├── data/
│   ├── download.py          # Bybit 데이터 다운로드
│   ├── preprocess.py         # 전처리 함수들
│   └── features.py           # Feature engineering
├── models/
│   ├── baseline.py           # Logistic, XGBoost
│   ├── deep_models.py        # CNN, LSTM, DeepLOB
│   └── utils.py
├── experiments/
│   ├── run_experiments.py    # 실험 실행
│   ├── configs.yaml          # 설정 파일
│   └── evaluate.py           # 평가
├── analysis/
│   ├── snr_analysis.py       # SNR 계산
│   ├── statistical_tests.py  # t-test, etc.
│   └── visualize.py          # Plots
├── notebooks/
│   ├── EDA.ipynb
│   └── Results.ipynb
└── requirements.txt
```

### **B. 핵심 전처리 코드**

```python
class LOBPreprocessor:
    def __init__(self, method='savgol'):
        self.method = method
        
    def denoise(self, mid_prices):
        if self.method == 'savgol':
            return savgol_filter(mid_prices, 11, 2)
        elif self.method == 'kalman':
            kf = KalmanFilter(...)
            return kf.filter(mid_prices)[0]
        elif self.method == 'wavelet':
            return self._wavelet_denoise(mid_prices)
        elif self.method == 'ma':
            return mid_prices.rolling(10).mean()
        else:  # raw
            return mid_prices
    
    def compute_snr(self, original, filtered):
        signal = filtered
        noise = original - filtered
        return 10 * np.log10(np.var(signal) / np.var(noise))
```

### **C. 실험 실행 예시**

```python
# experiments/run_experiments.py
import itertools

preprocessors = ['raw', 'savgol', 'kalman', 'wavelet', 'ma']
models = ['logistic', 'xgboost', 'catboost', 'cnn', 'deeplob', 'lstm']
depths = [5, 10, 20, 40]
horizons = [100, 500, 1000, 5000, 10000]  # ms

for prep, model, depth, horizon in itertools.product(...):
    # Load data
    data = load_lob_data(depth=depth)
    
    # Preprocess
    preprocessor = LOBPreprocessor(method=prep)
    data_preprocessed = preprocessor.denoise(data)
    
    # Extract features
    features = extract_features(data_preprocessed)
    
    # Train model
    clf = get_model(model)
    clf.fit(X_train, y_train)
    
    # Evaluate
    results = evaluate(clf, X_test, y_test)
    
    # Log results
    log_to_mlflow(prep, model, depth, horizon, results)
```

---

## **IX. 잠재적 문제 및 해결책**

### **Problem 1: 데이터 크기**
- **Issue**: LOB 데이터가 매우 큼 (GB 단위)
- **Solution**: 
  - Chunk-wise processing
  - HDF5 포맷 사용
  - 샘플링 (evenly-spaced)

### **Problem 2: Class Imbalance**
- **Issue**: Up/Down 비율 불균형
- **Solution**:
  - Stratified split
  - Class weights
  - SMOTE (신중하게)

### **Problem 3: 계산 시간**
- **Issue**: 600 configurations × long training
- **Solution**:
  - GPU 사용
  - Parallel processing (joblib)
  - Early stopping

### **Problem 4: 한국 데이터 접근**
- **Issue**: KOSPI LOB 데이터 유료/제한적
- **Solution**:
  - Crypto에 집중 (충분함)
  - 교수/학교 라이선스 활용
  - 또는 "future work"로 남김

---

## **X. 논문 구조 (Outline)**

### **Abstract** (150-200 words)
- Background: LOB prediction 중요성
- Problem: 모델 복잡도 vs 데이터 품질
- Method: 체계적 전처리 비교
- Results: 전처리가 1-2% 개선, 단순 모델 sufficient
- Contribution: 실무 가이드라인

### **1. Introduction**
- 1.1 Motivation
- 1.2 Research Gap
- 1.3 Research Questions
- 1.4 Contributions
- 1.5 Paper Structure

### **2. Literature Review**
- 2.1 Limit Order Book Microstructure
- 2.2 Machine Learning for LOB Prediction
- 2.3 Data Preprocessing in Finance
- 2.4 Signal Processing Techniques

### **3. Methodology**
- 3.1 Data Description
- 3.2 Preprocessing Methods
- 3.3 Feature Engineering
- 3.4 Models
- 3.5 Experimental Design
- 3.6 Evaluation Metrics

### **4. Results**
- 4.1 Overall Performance Comparison
- 4.2 Effect of Preprocessing by Asset Type
- 4.3 Effect by Model Complexity
- 4.4 Effect by Prediction Horizon
- 4.5 Signal-to-Noise Ratio Analysis
- 4.6 Computational Efficiency

### **5. Discussion**
- 5.1 Interpretation of Findings
- 5.2 When to Use Which Preprocessing?
- 5.3 Practical Guidelines
- 5.4 Limitations

### **6. Conclusion**
- 6.1 Summary
- 6.2 Contributions
- 6.3 Future Work

### **References**

### **Appendices**
- A. Hyperparameter Settings
- B. Additional Results
- C. Code Availability

---

## **XI. 핵심 인사이트 (Key Insights - 미리 준비)**

논문에서 강조할 메시지:

1. **"Simple is Better (with good data)"**
   - XGBoost + preprocessing ≥ DeepLOB (raw)
   - 실무에서는 단순 모델 선호 (해석 가능, 빠름)

2. **"Know Your Data"**
   - Illiquid assets → aggressive preprocessing
   - Liquid assets → minimal preprocessing
   - 자산 특성 파악이 선결 과제

3. **"One Size Doesn't Fit All"**
   - Horizon별로 최적 전처리 다름
   - Short horizon: Savitzky-Golay
   - Long horizon: Raw data sufficient

4. **"Latency Matters"**
   - Wavelet: 정확하지만 느림 → HFT 부적합
   - Savitzky-Golay: 빠르고 충분히 좋음 → 실용적

5. **"Signal-to-Noise is the Mechanism"**
   - 전처리 효과는 SNR 개선으로 설명 가능
   - 이론적 기반 제공

---

## **XII. 리스크 관리**

### **High Risk Items:**
1. **한국 데이터 접근 실패**
   - **Mitigation**: Crypto에만 집중 (충분함)

2. **결과가 기존 연구와 다름**
   - **Mitigation**: 차이 설명하는 섹션 추가, 재현성 강조

3. **계산 시간 초과**
   - **Mitigation**: 실험 범위 축소 (depth 2개만, horizon 3개만)

### **Medium Risk Items:**
4. **교수 반응 부정적**
   - **Mitigation**: 중간 발표로 피드백 조기 확보

5. **저널 리젝**
   - **Mitigation**: 국내 학회 먼저, 피드백 반영 후 저널

---

## **XIII. 성공 기준**

### **Minimum Viable Thesis (최소 요건):**
- ✅ Crypto LOB (BTC) 1개 자산
- ✅ 3개 전처리 방법
- ✅ 4개 모델
- ✅ 명확한 결과 (preprocessing helps)
- ✅ 졸업 통과

### **Good Thesis (목표):**
- ✅ Crypto 2개 자산 (BTC, ETH)
- ✅ 5개 전처리 방법
- ✅ 6개 모델
- ✅ SNR 분석 포함
- ✅ 국내 학회 발표

### **Excellent Thesis (이상적):**
- ✅ Crypto + 한국 주식
- ✅ 완전한 ablation study
- ✅ Trading simulation
- ✅ 오픈소스 코드
- ✅ International conference acceptance

---

## **XIV. 최종 체크리스트**

### **시작 전 (Week 0):**
- [ ] 교수 승인 확보
- [ ] GPU 서버 접근 확인
- [ ] 데이터 다운로드 테스트
- [ ] 환경 설정 완료

### **중간 점검 (Week 6):**
- [ ] Baseline 결과 확보
- [ ] 중간 발표 준비
- [ ] 교수 피드백 받기
- [ ] 일정 재조정

### **마무리 (Week 12):**
- [ ] 논문 초안 완성
- [ ] 코드 정리 및 문서화
- [ ] 발표 자료 준비
- [ ] 최종 제출

---

## **XV. 연락처 및 자료**

### **데이터 소스:**
- Bybit: https://www.bybit.com/derivatives/en/history-data
- LOBSTER: https://lobsterdata.com/
- LOBFrame: https://github.com/...

### **참고 코드:**
- DeepLOB: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- XGBoost: https://xgboost.readthedocs.io/

### **관련 논문:**
- Exploring Microstructural Dynamics (2025)
- Deep Limit Order Book Forecasting (2024)
- Feature Engineering for Mid-Price Prediction (2019)

---

# **완료. 이제 시작하세요!**

**질문이나 막히는 부분 있으면 언제든 물어보세요. 당신은 할 수 있습니다. 화이팅!** 🔥