# LOB Mid-Price Prediction: Preprocessing vs Feature Engineering

LOB(Limit Order Book) 데이터에서 전처리(preprocessing)와 피처 엔지니어링(feature engineering)을 체계적으로 비교한 연구 프로젝트.

**핵심 결론**: 전처리는 이미 정규화된 데이터에선 무의미하고, raw LOB + 도메인 피처 조합이 유의미한 성능 향상을 만든다.

## 주요 결과

| 구성 | 정확도 (%) | 향상폭 | p-value |
|------|-----------|--------|---------|
| Raw baseline | 68.47 ± 0.39 | - | - |
| Raw + Engineered (78 features) | **73.43 ± 0.33** | **+4.96 pp** | **< 0.001** |
| TransLOB (raw+eng) | 67.15 ± 0.54 | - | - |

- CatBoost가 TransLOB 대비 **+6.28 pp** 우위 (tabular 데이터에선 gradient boosting이 Transformer보다 효과적)
- 도메인 지식이 단순 차원 증가 대비 **60% 더 기여** (random feature baseline 대비 +3.88 pp)
- KRX 한국 주식 9종 cross-market 검증: **77.65% ± 0.20%** (p = 1.04 × 10⁻¹⁰)

## 38개 엔지니어링 피처

| 그룹 | 개수 | 설명 |
|------|------|------|
| Price | 6 | mid-price, VWAP, spread, volatility |
| Volume | 8 | bid/ask volume ratio, 누적 volume |
| Order Imbalance (OI) | 6 | 수급 비대칭 |
| Order Flow Imbalance (OFI) | 6 | 주문 흐름 변화 (Cont et al., 2014) |
| Depth | 6 | depth imbalance, liquidity concentration |
| Price Impact | 6 | market order impact 추정 |

## 실시간 추론

스트리밍 파이프라인으로 78개 피처를 이벤트 단위 실시간 계산:

- **20.3μs/event** (P99: 41.8μs) — 200μs 목표 대비 10배 여유
- **49,285 events/sec** throughput
- batch 파이프라인과 수치적으로 동일 (max diff: 1.46 × 10⁻¹¹)

## 데이터셋

- **FI-2010**: NASDAQ Nordic 5종목, ~4M samples (Z-score 정규화 벤치마크)
- **KRX**: 한국투자증권 API로 수집한 KOSPI/KOSDAQ 9종목, 4.38M snapshots

## 프로젝트 구조

```
├── lob_preprocessing/    # 전처리 (wavelet, Kalman, MA)
├── feature_engineering/  # 38개 도메인 피처 계산
├── model_training/       # CatBoost, TransLOB 학습
├── lob_realtime/         # 스트리밍 추론 파이프라인
├── validation/           # data leakage 검증
├── data/                 # FI-2010, KRX 데이터
├── results/              # 실험 결과
└── PAPER_DRAFT.md        # 논문 초안
```

## 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 검증 포인트

- 5 random seeds (42, 123, 456, 789, 1011) paired t-test
- temporal train/test split (look-ahead bias 없음)
- 전체 피처 causality 검증 완료
- 정규화는 train set에서만 fit

## License

MIT
