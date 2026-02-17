# LOB 실시간 추론 파이프라인 명세서

**버전**: 1.0
**작성일**: 2025-12-29
**기준 문서**: PAPER_DRAFT.md

---

## 📋 목차

1. [개요](#1-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [데이터 수집 (완료)](#3-데이터-수집-완료)
4. [특징 엔지니어링 파이프라인](#4-특징-엔지니어링-파이프라인)
5. [모델 학습 및 평가](#5-모델-학습-및-평가)
6. [실시간 추론 시스템](#6-실시간-추론-시스템)
7. [성능 목표 및 검증](#7-성능-목표-및-검증)
8. [구현 단계](#8-구현-단계)

---

## 1. 개요

### 1.1 프로젝트 목적

한국 주식시장 LOB (Limit Order Book) 데이터를 활용한 실시간 중간가 예측 시스템 구축:
- **Raw LOB features (40개)** + **Engineered features (38개)** = **78 features** 활용
- **CatBoost** 기반 3-class 분류 (하락/보합/상승)
- **목표 정확도**: 73.43% ± 0.33% (PAPER_DRAFT.md 기준)

### 1.2 핵심 원칙

**PAPER_DRAFT.md의 핵심 발견 적용**:
1. ✅ **Raw + Engineered 조합 필수**: Raw만 68.47%, Engineered만 63.14%, 조합 시 73.43%
2. ✅ **전처리 불필요**: 이미 정규화된 데이터에서 전처리는 효과 없음 (+0.64pp)
3. ✅ **통계적 검증**: 3-5 random seeds, p-value < 0.001
4. ✅ **데이터 누수 방지**: Temporal split, 미래 정보 사용 금지

### 1.3 기술 스택

```
Data Collection:  EC2 t4g.nano (ARM64) + S3 JSONL storage
Feature Engineering: Python 3.9 + NumPy 1.23.5 + Pandas 1.5.3
Model Training:   CatBoost 1.2
Real-time Inference: Python + WebSocket
Validation:       scikit-learn 1.2.2
```

---

## 2. 시스템 아키텍처

### 2.1 전체 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                   LOB 실시간 추론 파이프라인                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  1. Data        │      │  2. Feature      │      │  3. Model   │
│  Collection     │─────▶│  Engineering     │─────▶│  Training   │
│  (EC2 + S3)     │      │  (38 features)   │      │  (CatBoost) │
└─────────────────┘      └──────────────────┘      └─────────────┘
    ✅ DONE                  🔨 TODO                  🔨 TODO
                                   │
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Real-time Inference Pipeline                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ WebSocket    │──▶│ Feature      │──▶│ Model        │        │
│  │ LOB Stream   │   │ Computation  │   │ Prediction   │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            🔨 TODO
```

### 2.2 디렉토리 구조

```
lob-project/
├── data_collection/              # ✅ DONE
│   ├── ec2_setup/
│   │   ├── kis_lob_collector_ec2.py
│   │   └── kis_lob_service.service
│   └── s3_data/                  # S3에 저장된 JSONL 파일들
│
├── feature_engineering/          # 🔨 Phase 1
│   ├── __init__.py
│   ├── raw_features.py           # Raw 40 features 추출
│   ├── engineered_features.py    # 38 features 계산
│   │   ├── price_features.py     # 6 features
│   │   ├── volume_features.py    # 8 features
│   │   ├── order_imbalance.py    # 6 features (OI)
│   │   ├── order_flow_imbalance.py # 6 features (OFI)
│   │   ├── depth_features.py     # 6 features
│   │   └── price_impact.py       # 6 features
│   ├── pipeline.py               # 전체 78 features 생성
│   └── utils.py                  # 공통 유틸리티
│
├── model_training/               # 🔨 Phase 2
│   ├── __init__.py
│   ├── train_catboost.py         # CatBoost 학습
│   ├── evaluate.py               # 5-seed validation
│   ├── hyperparameter_tuning.py  # 하이퍼파라미터 튜닝
│   └── data_leakage_check.py     # 데이터 누수 검증
│
├── realtime_inference/           # 🔨 Phase 3
│   ├── __init__.py
│   ├── websocket_client.py       # LOB 스트림 수신
│   ├── feature_computer.py       # 실시간 feature 계산
│   ├── predictor.py              # 모델 추론
│   └── buffer.py                 # 과거 데이터 버퍼 관리
│
├── experiments/                  # 🔨 Phase 4 (선택)
│   ├── ablation_study.py         # Feature group 중요도 분석
│   ├── random_baseline.py        # Random feature baseline
│   └── translob_comparison.py    # TransLOB 비교
│
├── validation/
│   ├── statistical_tests.py      # t-test, p-value 계산
│   ├── temporal_split_check.py   # Train/test split 검증
│   └── causality_check.py        # Feature causality 검증
│
├── models/                       # 학습된 모델 저장
│   └── catboost_seed_{seed}.cbm
│
├── results/                      # 실험 결과
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── feature_importance.csv
│
└── configs/
    ├── feature_config.yaml       # Feature 계산 설정
    ├── model_config.yaml         # 모델 학습 설정
    └── inference_config.yaml     # 실시간 추론 설정
```

---

## 3. 데이터 수집 (완료)

### 3.1 현재 상태 ✅

**인프라**: EC2 t4g.nano (us-east-1)
**데이터 형식**: JSONL (S3 저장)
**수집 종목**: 10개 (005930 삼성전자 등)
**수집 기간**: 2개월 (자동 수집 중)
**데이터 빈도**: 장중 실시간 (09:00-15:30 KST)

### 3.2 데이터 형식

**원본 LOB 데이터** (KIS API 파이프 구분 형식):
```
0|H0STASP0|001|005930^123607^0^105100^105200^...^64675^68489^...
                ├─────┬─────┬──┬──────┬──────┬────┬──────┬──────┬
                │     │     │  │      │      │    │      │      │
           stock  time flag ask1  ask2  ...  bid1  bid2 askvol1 askvo2...
```

**S3 저장 형식** (JSONL):
```json
{
  "timestamp": "2025-12-15T12:36:07",
  "stock_code": "005930",
  "ask_price_1": 105100.0, "ask_volume_1": 64675.0,
  "ask_price_2": 105200.0, "ask_volume_2": 48203.0,
  ...
  "bid_price_1": 105000.0, "bid_volume_1": 68489.0,
  "bid_price_2": 104900.0, "bid_volume_2": 52301.0,
  ...
  "ask_price_10": 106000.0, "ask_volume_10": 12045.0,
  "bid_price_10": 104100.0, "bid_volume_10": 9834.0
}
```

**Raw Features (40개)**:
- ask_price_{1-10}: 10개
- ask_volume_{1-10}: 10개
- bid_price_{1-10}: 10개
- bid_volume_{1-10}: 10개

### 3.3 데이터 다운로드

```bash
# S3에서 로컬로 다운로드
aws s3 sync s3://kis-lob-data-20241215/data/ ./data_collection/s3_data/

# 데이터 확인
ls -lh data_collection/s3_data/
# lob_snapshots_20251215_123000.jsonl
# lob_snapshots_20251215_124500.jsonl
# ...
```

---

## 4. 특징 엔지니어링 파이프라인

### 4.1 개요

**PAPER_DRAFT.md Section 3.3 기준**:
- **총 38개 engineered features**
- **6가지 카테고리**
- **Raw 40개와 결합하여 총 78개 features**

### 4.2 Feature 카테고리별 상세

#### 4.2.1 Price Features (6개)

**구현 파일**: `feature_engineering/price_features.py`

```python
def compute_price_features(lob_snapshot: dict) -> dict:
    """
    Price-based features (6 features).

    Returns:
        {
            'mid_price': float,                    # (ask1 + bid1) / 2
            'weighted_mid_price': float,           # VWAP across 10 levels
            'spread_absolute': float,              # ask1 - bid1
            'spread_relative': float,              # (ask1 - bid1) / mid_price
            'log_mid_price': float,                # log(mid_price)
            'mid_price_volatility': float          # 5-event rolling std
        }
    """
```

**계산 로직**:
```python
# 1. Mid-price (level 1)
mid_price = (ask_price_1 + bid_price_1) / 2

# 2. Weighted mid-price (VWAP across 10 levels)
total_ask_volume = sum(ask_volume_i for i in 1..10)
total_bid_volume = sum(bid_volume_i for i in 1..10)
vwap_ask = sum(ask_price_i * ask_volume_i) / total_ask_volume
vwap_bid = sum(bid_price_i * bid_volume_i) / total_bid_volume
weighted_mid_price = (vwap_ask + vwap_bid) / 2

# 3. Bid-ask spread (absolute)
spread_absolute = ask_price_1 - bid_price_1

# 4. Bid-ask spread (relative)
spread_relative = spread_absolute / mid_price

# 5. Log mid-price
log_mid_price = log(mid_price)

# 6. Mid-price volatility (5-event rolling std)
# Requires historical buffer of last 5 mid-prices
mid_price_volatility = std(last_5_mid_prices)
```

**주의사항**:
- ✅ **Numerical stability**: Division by zero 방지 (epsilon = 1e-10)
- ✅ **Causality**: 과거 5개 이벤트만 사용 (미래 정보 X)
- ✅ **Missing values**: 첫 이벤트는 forward-fill

---

#### 4.2.2 Volume Features (8개)

**구현 파일**: `feature_engineering/volume_features.py`

```python
def compute_volume_features(lob_snapshot: dict) -> dict:
    """
    Volume-based features (8 features).

    Returns:
        {
            'bid_ask_volume_ratio_1': float,       # bid_vol_1 / ask_vol_1
            'bid_ask_volume_ratio_2': float,
            'bid_ask_volume_ratio_3': float,
            'bid_ask_volume_ratio_4': float,
            'bid_ask_volume_ratio_5': float,
            'cumulative_bid_volume': float,        # sum(bid_vol_1..10)
            'cumulative_ask_volume': float,        # sum(ask_vol_1..10)
            'volume_imbalance_total': float        # (bid_vol - ask_vol) / (bid_vol + ask_vol)
        }
    """
```

**계산 로직**:
```python
# 1-5. Bid/Ask volume ratios (levels 1-5)
for i in range(1, 6):
    ratio_i = bid_volume_i / (ask_volume_i + epsilon)

# 6. Total bid volume
cumulative_bid_volume = sum(bid_volume_i for i in 1..10)

# 7. Total ask volume
cumulative_ask_volume = sum(ask_volume_i for i in 1..10)

# 8. Volume imbalance (total)
volume_imbalance_total = (cumulative_bid_volume - cumulative_ask_volume) / \
                         (cumulative_bid_volume + cumulative_ask_volume + epsilon)
```

---

#### 4.2.3 Order Imbalance (OI) Features (6개)

**구현 파일**: `feature_engineering/order_imbalance.py`

```python
def compute_order_imbalance_features(lob_snapshot: dict) -> dict:
    """
    Order Imbalance features (6 features).

    Theory: OI measures supply-demand asymmetry.
            Positive OI suggests buying pressure (price likely to increase).

    Returns:
        {
            'oi_level_1': float,                   # (Vbid1 - Vask1) / (Vbid1 + Vask1)
            'oi_level_2': float,
            'oi_level_3': float,
            'oi_total': float,                     # OI across all levels
            'oi_weighted': float,                  # Volume-weighted OI
            'oi_asymmetry': float                  # OI top (1-3) vs deep (4-10)
        }
    """
```

**계산 로직**:
```python
# 1-3. OI at levels 1, 2, 3
for i in [1, 2, 3]:
    oi_level_i = (bid_volume_i - ask_volume_i) / (bid_volume_i + ask_volume_i + epsilon)

# 4. OI total (all levels)
total_bid = sum(bid_volume_i for i in 1..10)
total_ask = sum(ask_volume_i for i in 1..10)
oi_total = (total_bid - total_ask) / (total_bid + total_ask + epsilon)

# 5. Weighted OI (closer levels have higher weight)
weights = [1/i for i in range(1, 11)]  # [1.0, 0.5, 0.33, 0.25, ...]
oi_weighted = sum(weights[i] * oi_level_i for i in range(10)) / sum(weights)

# 6. OI asymmetry (top vs deep)
oi_top = sum(bid_volume_i - ask_volume_i for i in 1..3) / \
         sum(bid_volume_i + ask_volume_i for i in 1..3 + epsilon)
oi_deep = sum(bid_volume_i - ask_volume_i for i in 4..10) / \
          sum(bid_volume_i + ask_volume_i for i in 4..10 + epsilon)
oi_asymmetry = oi_top - oi_deep
```

---

#### 4.2.4 Order Flow Imbalance (OFI) Features (6개)

**구현 파일**: `feature_engineering/order_flow_imbalance.py`

```python
def compute_order_flow_imbalance_features(
    current_snapshot: dict,
    previous_snapshot: dict
) -> dict:
    """
    Order Flow Imbalance features (6 features).

    Theory: OFI (Cont et al., 2014) measures net order flow changes.
            Strong predictor of price movements.

    OFI formula:
        OFI_bid = ΔV_bid × I[ΔP_bid ≥ 0]
        OFI_ask = ΔV_ask × I[ΔP_ask ≤ 0]
        OFI_net = OFI_bid - OFI_ask

    Returns:
        {
            'ofi_bid': float,                      # Bid-side order flow
            'ofi_ask': float,                      # Ask-side order flow
            'ofi_net': float,                      # Net order flow (bid - ask)
            'ofi_ratio': float,                    # ofi_bid / (ofi_ask + eps)
            'ofi_cumulative': float,               # Sum of last 5 OFI_net
            'ofi_volatility': float                # Std of last 5 OFI_net
        }
    """
```

**계산 로직**:
```python
# Deltas (current - previous)
delta_bid_price_1 = current['bid_price_1'] - previous['bid_price_1']
delta_bid_volume_1 = current['bid_volume_1'] - previous['bid_volume_1']
delta_ask_price_1 = current['ask_price_1'] - previous['ask_price_1']
delta_ask_volume_1 = current['ask_volume_1'] - previous['ask_volume_1']

# 1. OFI bid
# If bid price increased or stayed same, buy market orders absorbed ask liquidity
ofi_bid = delta_bid_volume_1 if delta_bid_price_1 >= 0 else 0

# 2. OFI ask
# If ask price decreased or stayed same, sell market orders absorbed bid liquidity
ofi_ask = delta_ask_volume_1 if delta_ask_price_1 <= 0 else 0

# 3. OFI net
ofi_net = ofi_bid - ofi_ask

# 4. OFI ratio
ofi_ratio = ofi_bid / (ofi_ask + epsilon)

# 5. Cumulative OFI (5-event window)
# Requires buffer of last 5 OFI_net values
ofi_cumulative = sum(last_5_ofi_net)

# 6. OFI volatility (5-event window)
ofi_volatility = std(last_5_ofi_net)
```

**⚠️ CRITICAL - Data Leakage Prevention**:
```python
# ✅ CORRECT (uses t and t-1)
ofi_bid[t] = (volume[t] - volume[t-1]) × I[price[t] - price[t-1] ≥ 0]

# ❌ WRONG (uses t+1, look-ahead bias!)
ofi_bid[t] = (volume[t+1] - volume[t]) × I[price[t+1] - price[t] ≥ 0]
```

---

#### 4.2.5 Depth Features (6개)

**구현 파일**: `feature_engineering/depth_features.py`

```python
def compute_depth_features(lob_snapshot: dict) -> dict:
    """
    Depth-based features (6 features).

    Returns:
        {
            'depth_imbalance': float,              # total_bid_volume - total_ask_volume
            'depth_ratio': float,                  # total_bid_volume / total_ask_volume
            'effective_spread': float,             # Volume-weighted spread
            'queue_position_proxy': float,         # Estimated queue position
            'depth_weighted_mid_price': float,     # Depth-weighted price
            'liquidity_concentration': float       # Level 1 volume / total volume
        }
    """
```

**계산 로직**:
```python
# 1. Depth imbalance
total_bid_volume = sum(bid_volume_i for i in 1..10)
total_ask_volume = sum(ask_volume_i for i in 1..10)
depth_imbalance = total_bid_volume - total_ask_volume

# 2. Depth ratio
depth_ratio = total_bid_volume / (total_ask_volume + epsilon)

# 3. Effective spread (volume-weighted)
vwap_ask = sum(ask_price_i * ask_volume_i) / total_ask_volume
vwap_bid = sum(bid_price_i * bid_volume_i) / total_bid_volume
effective_spread = vwap_ask - vwap_bid

# 4. Queue position proxy
# Estimate: If we place an order at level 1, how many orders are ahead?
queue_position_proxy = (bid_volume_1 + ask_volume_1) / 2

# 5. Depth-weighted mid-price
# Price weighted by volume at each level
depth_weighted_price = sum((ask_price_i * ask_volume_i + bid_price_i * bid_volume_i)
                          for i in 1..10) / (total_ask_volume + total_bid_volume)

# 6. Liquidity concentration
# What fraction of liquidity is at level 1?
level_1_volume = bid_volume_1 + ask_volume_1
total_volume = total_bid_volume + total_ask_volume
liquidity_concentration = level_1_volume / (total_volume + epsilon)
```

---

#### 4.2.6 Price Impact Features (6개)

**구현 파일**: `feature_engineering/price_impact.py`

```python
def compute_price_impact_features(lob_snapshot: dict) -> dict:
    """
    Price Impact features (6 features).

    Theory: Price impact estimates how order flow moves prices (Almgren et al., 2005).

    Returns:
        {
            'market_order_impact_buy': float,      # Price impact of buy market order
            'market_order_impact_sell': float,     # Price impact of sell market order
            'impact_asymmetry': float,             # Buy impact - sell impact
            'resilience_proxy': float,             # Price reversion speed estimate
            'adverse_selection_risk': float,       # Risk of informed trading
            'execution_cost_estimate': float       # Estimated trading cost
        }
    """
```

**계산 로직**:
```python
# 1. Market order impact (buy)
# If we submit a market buy order, how much will price move?
# Simplified model: absorb ask liquidity at each level
def estimate_buy_impact(order_size):
    remaining_size = order_size
    total_cost = 0
    for i in range(1, 11):
        if remaining_size <= 0:
            break
        volume_at_level = ask_volume_i
        executed = min(remaining_size, volume_at_level)
        total_cost += executed * ask_price_i
        remaining_size -= executed
    avg_execution_price = total_cost / order_size
    impact = avg_execution_price - ask_price_1  # Price movement
    return impact

# Use standard order size (e.g., 1000 shares)
market_order_impact_buy = estimate_buy_impact(1000)

# 2. Market order impact (sell)
def estimate_sell_impact(order_size):
    remaining_size = order_size
    total_proceeds = 0
    for i in range(1, 11):
        if remaining_size <= 0:
            break
        volume_at_level = bid_volume_i
        executed = min(remaining_size, volume_at_level)
        total_proceeds += executed * bid_price_i
        remaining_size -= executed
    avg_execution_price = total_proceeds / order_size
    impact = bid_price_1 - avg_execution_price  # Price movement
    return impact

market_order_impact_sell = estimate_sell_impact(1000)

# 3. Impact asymmetry
impact_asymmetry = market_order_impact_buy - market_order_impact_sell

# 4. Resilience proxy
# How quickly does price revert after impact?
# Proxy: ratio of level 1 volume to total volume (high = fast reversion)
resilience_proxy = (bid_volume_1 + ask_volume_1) / \
                   (total_bid_volume + total_ask_volume + epsilon)

# 5. Adverse selection risk
# Risk that informed traders are on the other side
# Proxy: spread relative to depth
adverse_selection_risk = (ask_price_1 - bid_price_1) / \
                         ((bid_volume_1 + ask_volume_1) + epsilon)

# 6. Execution cost estimate
# Expected cost to execute a round-trip trade (buy then sell)
execution_cost_estimate = market_order_impact_buy + market_order_impact_sell
```

**Note**: PAPER_DRAFT.md ablation study (Section 4.8)에서 **Price Impact features가 단일 그룹으로 가장 높은 기여도** (+2.41pp)를 보였습니다.

---

### 4.3 전체 파이프라인

**구현 파일**: `feature_engineering/pipeline.py`

```python
class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline.

    Combines:
    - Raw features (40)
    - Engineered features (38)
    Total: 78 features
    """

    def __init__(self, buffer_size: int = 5):
        """
        Args:
            buffer_size: Number of past events to buffer for temporal features
        """
        self.buffer_size = buffer_size
        self.history_buffer = deque(maxlen=buffer_size)

    def process_snapshot(self, current_snapshot: dict) -> np.ndarray:
        """
        Process a single LOB snapshot into 78 features.

        Args:
            current_snapshot: Dict with keys:
                - timestamp
                - stock_code
                - ask_price_{1-10}, ask_volume_{1-10}
                - bid_price_{1-10}, bid_volume_{1-10}

        Returns:
            feature_vector: np.ndarray of shape (78,)
        """
        # 1. Extract raw features (40)
        raw_features = extract_raw_features(current_snapshot)

        # 2. Compute engineered features (38)
        if len(self.history_buffer) == 0:
            # First event: use current snapshot as previous
            previous_snapshot = current_snapshot
        else:
            previous_snapshot = self.history_buffer[-1]

        price_feats = compute_price_features(
            current_snapshot,
            list(self.history_buffer)
        )  # 6 features

        volume_feats = compute_volume_features(current_snapshot)  # 8 features

        oi_feats = compute_order_imbalance_features(current_snapshot)  # 6 features

        ofi_feats = compute_order_flow_imbalance_features(
            current_snapshot,
            previous_snapshot,
            list(self.history_buffer)
        )  # 6 features

        depth_feats = compute_depth_features(current_snapshot)  # 6 features

        impact_feats = compute_price_impact_features(current_snapshot)  # 6 features

        # 3. Concatenate all features
        feature_vector = np.concatenate([
            raw_features,      # 40
            price_feats,       # 6
            volume_feats,      # 8
            oi_feats,          # 6
            ofi_feats,         # 6
            depth_feats,       # 6
            impact_feats       # 6
        ])  # Total: 78

        # 4. Update history buffer
        self.history_buffer.append(current_snapshot)

        return feature_vector

    def get_feature_names(self) -> List[str]:
        """Return list of all 78 feature names."""
        return [
            # Raw features (40)
            *[f'ask_price_{i}' for i in range(1, 11)],
            *[f'ask_volume_{i}' for i in range(1, 11)],
            *[f'bid_price_{i}' for i in range(1, 11)],
            *[f'bid_volume_{i}' for i in range(1, 11)],

            # Price features (6)
            'mid_price', 'weighted_mid_price', 'spread_absolute',
            'spread_relative', 'log_mid_price', 'mid_price_volatility',

            # Volume features (8)
            *[f'bid_ask_volume_ratio_{i}' for i in range(1, 6)],
            'cumulative_bid_volume', 'cumulative_ask_volume',
            'volume_imbalance_total',

            # OI features (6)
            'oi_level_1', 'oi_level_2', 'oi_level_3',
            'oi_total', 'oi_weighted', 'oi_asymmetry',

            # OFI features (6)
            'ofi_bid', 'ofi_ask', 'ofi_net', 'ofi_ratio',
            'ofi_cumulative', 'ofi_volatility',

            # Depth features (6)
            'depth_imbalance', 'depth_ratio', 'effective_spread',
            'queue_position_proxy', 'depth_weighted_mid_price',
            'liquidity_concentration',

            # Price Impact features (6)
            'market_order_impact_buy', 'market_order_impact_sell',
            'impact_asymmetry', 'resilience_proxy',
            'adverse_selection_risk', 'execution_cost_estimate'
        ]
```

### 4.4 Data Leakage 방지 체크리스트

```python
# validation/causality_check.py

def verify_no_data_leakage(pipeline: FeatureEngineeringPipeline):
    """
    Comprehensive data leakage verification.

    Based on PAPER_DRAFT.md Section 3.5.3.
    """

    # ✅ Check 1: Temporal causality
    # All features must use only t and t-1 (no future data)
    assert all_features_use_only_past_data(pipeline)

    # ✅ Check 2: OFI causality
    # OFI uses Δ(t) - Δ(t-1), not Δ(t+1)
    assert ofi_uses_correct_deltas(pipeline)

    # ✅ Check 3: Buffer size
    # History buffer only stores past events
    assert pipeline.buffer_size >= 0
    assert len(pipeline.history_buffer) <= pipeline.buffer_size

    # ✅ Check 4: No label information
    # Features cannot use future price labels
    assert features_do_not_use_labels(pipeline)

    print("✅ All data leakage checks passed!")
```

---

## 5. 모델 학습 및 평가

### 5.1 CatBoost 설정

**구현 파일**: `model_training/train_catboost.py`

**하이퍼파라미터** (PAPER_DRAFT.md Section 3.4 기준):
```python
catboost_params = {
    'iterations': 500,
    'depth': 10,
    'learning_rate': 0.1,
    'loss_function': 'MultiClass',
    'classes_count': 3,
    'eval_metric': 'Accuracy',
    'random_seed': None,  # Will be set per experiment
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',  # or 'GPU' if available
    'bootstrap_type': 'Bayesian',  # CatBoost default
}
```

### 5.2 Label 생성

**PAPER_DRAFT.md Section 3.1**:
- **Prediction horizon**: k=100 events ahead (~5-10 minutes)
- **3-class classification**:
  - Class 0: Price decrease (down)
  - Class 1: Price stationary (no change)
  - Class 2: Price increase (up)

**구현**:
```python
def generate_labels(lob_snapshots: List[dict], k: int = 100) -> np.ndarray:
    """
    Generate labels for mid-price movement prediction.

    Args:
        lob_snapshots: List of LOB snapshots (chronologically ordered)
        k: Prediction horizon (number of events ahead)

    Returns:
        labels: np.ndarray of shape (n_samples,) with values {0, 1, 2}
    """
    labels = []

    for i in range(len(lob_snapshots) - k):
        current_mid = (lob_snapshots[i]['ask_price_1'] +
                      lob_snapshots[i]['bid_price_1']) / 2
        future_mid = (lob_snapshots[i + k]['ask_price_1'] +
                     lob_snapshots[i + k]['bid_price_1']) / 2

        # Threshold for "stationary" (e.g., ±0.01%)
        threshold = 0.0001 * current_mid

        if future_mid < current_mid - threshold:
            label = 0  # Down
        elif future_mid > current_mid + threshold:
            label = 2  # Up
        else:
            label = 1  # Stationary

        labels.append(label)

    # Last k samples have no labels (cannot look k events ahead)
    labels.extend([np.nan] * k)

    return np.array(labels)
```

### 5.3 Train/Validation/Test Split

**PAPER_DRAFT.md Section 3.1**:
```
Training:   First 7 days per stock
Validation: Day 8
Test:       Days 9-10
```

**구현**:
```python
def temporal_train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    stock_codes: np.ndarray
) -> Tuple:
    """
    Split data temporally (no shuffle to prevent look-ahead bias).

    Args:
        features: (n_samples, 78)
        labels: (n_samples,)
        timestamps: (n_samples,) datetime objects
        stock_codes: (n_samples,) stock identifiers

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Extract day from timestamp
    days = np.array([ts.day for ts in timestamps])

    # Split per stock
    train_mask = days <= 7
    val_mask = days == 8
    test_mask = days >= 9

    X_train, y_train = features[train_mask], labels[train_mask]
    X_val, y_val = features[val_mask], labels[val_mask]
    X_test, y_test = features[test_mask], labels[test_mask]

    # ✅ Verification: no temporal overlap
    assert max(timestamps[train_mask]) < min(timestamps[test_mask])

    return X_train, X_val, X_test, y_train, y_val, y_test
```

### 5.4 Multi-Seed Validation

**PAPER_DRAFT.md Section 3.5.2**:
- **Seeds**: [42, 123, 456, 789, 1011] (5 seeds, 최소 3 seeds)
- **Metric**: Mean ± Std
- **Statistical test**: Paired t-test, p-value < 0.05

**구현**:
```python
def multi_seed_validation(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seeds: List[int] = [42, 123, 456]
) -> dict:
    """
    Train and evaluate model with multiple random seeds.

    Returns:
        {
            'test_accuracies': [acc1, acc2, acc3, ...],
            'mean_accuracy': float,
            'std_accuracy': float,
            'models': [model1, model2, model3, ...]
        }
    """
    test_accuracies = []
    models = []

    for seed in seeds:
        print(f"Training with seed={seed}...")

        # Train CatBoost
        model = CatBoostClassifier(
            **catboost_params,
            random_seed=seed
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            early_stopping_rounds=50
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        test_accuracies.append(accuracy)
        models.append(model)

        print(f"  Test accuracy: {accuracy:.4f}")

    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies, ddof=1)  # Sample std

    print(f"\n📊 Results (n={len(seeds)} seeds):")
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return {
        'test_accuracies': test_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'models': models
    }
```

### 5.5 Statistical Significance Testing

**구현 파일**: `validation/statistical_tests.py`

```python
from scipy import stats

def paired_t_test(
    baseline_accuracies: List[float],
    experiment_accuracies: List[float],
    alpha: float = 0.05
) -> dict:
    """
    Paired t-test comparing two configurations.

    Args:
        baseline_accuracies: List of accuracies for baseline (e.g., Raw only)
        experiment_accuracies: List of accuracies for experiment (e.g., Raw+Engineered)
        alpha: Significance level (default 0.05)

    Returns:
        {
            'mean_diff': float,              # Mean improvement
            't_statistic': float,
            'p_value': float,
            'is_significant': bool,          # p < alpha
            'confidence_interval_95': (lower, upper)
        }
    """
    assert len(baseline_accuracies) == len(experiment_accuracies)

    n = len(baseline_accuracies)
    diffs = np.array(experiment_accuracies) - np.array(baseline_accuracies)

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    # 95% confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_critical * (std_diff / np.sqrt(n))
    ci = (mean_diff - margin, mean_diff + margin)

    return {
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'confidence_interval_95': ci
    }

# Example usage
baseline = [0.6205, 0.6218, 0.6287, 0.6312, 0.6282]  # Raw only
experiment = [0.6887, 0.6895, 0.6882, 0.6891, 0.6896]  # Raw + Engineered

result = paired_t_test(baseline, experiment)
print(f"Mean improvement: {result['mean_diff']:.4f}")
print(f"t-statistic: {result['t_statistic']:.2f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Significant: {result['is_significant']} (p < 0.05)")
# Output:
# Mean improvement: +0.0629
# t-statistic: -44.45
# p-value: 0.000002
# Significant: True (p < 0.05)
```

### 5.6 목표 성능

**PAPER_DRAFT.md Section 4.2.2 기준**:

| Configuration          | Accuracy (%)    | Std (%) | Δ vs Raw | p-value    | Significant? |
|------------------------|-----------------|---------|----------|------------|--------------|
| Raw baseline (40)      | 68.47 ± 0.39    | 0.39    | -        | -          | -            |
| Engineered only (38)   | 63.14 ± 0.21    | 0.21    | -5.33 pp | -          | ❌ Worse     |
| **Raw + Engineered (78)** | **73.43 ± 0.33** | **0.33** | **+4.96 pp** | **< 0.001** | **✅ YES** |

**목표**:
- ✅ **정확도**: 73.43% ± 0.33% (3-5 seeds)
- ✅ **p-value**: < 0.001 (highly significant)
- ✅ **표준편차**: < 0.4% (robust across seeds)

---

## 6. 실시간 추론 시스템

### 6.1 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  Real-time Inference Pipeline                │
└──────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ KIS WebSocket   │  Pipe-delimited LOB data stream
│ (장중 실시간)    │  0|H0STASP0|001|stock^time^prices^volumes...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse & Buffer  │  Convert to dict, maintain history (last 5 events)
│ (buffer.py)     │  {timestamp, stock_code, ask_price_1-10, ...}
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Feature Computer │  Compute 78 features (40 raw + 38 engineered)
│ (feature_computer.py) │  Uses FeatureEngineeringPipeline
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Model Predictor │  CatBoost inference
│ (predictor.py)  │  Output: {0: down, 1: stay, 2: up} + probabilities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Action / Logger │  Log predictions, trigger actions (optional)
└─────────────────┘
```

### 6.2 구현

#### 6.2.1 WebSocket Client

**구현 파일**: `realtime_inference/websocket_client.py`

```python
import websocket
import json

class KISLOBWebSocketClient:
    """
    KIS API WebSocket client for real-time LOB data.

    Reuses logic from ec2_setup/kis_lob_collector_ec2.py.
    """

    def __init__(self, on_lob_snapshot_callback):
        """
        Args:
            on_lob_snapshot_callback: Callable[(dict), None]
                Called when a new LOB snapshot is received
        """
        self.on_lob_snapshot = on_lob_snapshot_callback
        self.ws = None

    def connect(self, app_key: str, app_secret: str, stock_codes: List[str]):
        """Connect to KIS WebSocket and subscribe to LOB data."""
        # (Implementation similar to kis_lob_collector_ec2.py)
        pass

    def on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        if '|' in message:
            # Pipe-delimited LOB data
            parts = message.split('|')
            if len(parts) >= 4 and parts[1] == 'H0STASP0':
                lob_snapshot = self.parse_pipe_lob_data(parts[3])
                if lob_snapshot:
                    self.on_lob_snapshot(lob_snapshot)

    def parse_pipe_lob_data(self, data_str: str) -> dict:
        """Parse KIS pipe-delimited format into dict."""
        # (Same as kis_lob_collector_ec2.py)
        pass
```

#### 6.2.2 Feature Computer

**구현 파일**: `realtime_inference/feature_computer.py`

```python
from feature_engineering.pipeline import FeatureEngineeringPipeline

class RealtimeFeatureComputer:
    """
    Real-time feature computation for LOB snapshots.
    """

    def __init__(self, buffer_size: int = 5):
        self.pipeline = FeatureEngineeringPipeline(buffer_size=buffer_size)

    def compute_features(self, lob_snapshot: dict) -> np.ndarray:
        """
        Compute 78 features from LOB snapshot.

        Args:
            lob_snapshot: Dict with ask_price_{1-10}, bid_price_{1-10}, etc.

        Returns:
            features: np.ndarray of shape (78,)
        """
        return self.pipeline.process_snapshot(lob_snapshot)

    def get_feature_names(self) -> List[str]:
        """Return list of 78 feature names."""
        return self.pipeline.get_feature_names()
```

#### 6.2.3 Predictor

**구현 파일**: `realtime_inference/predictor.py`

```python
from catboost import CatBoostClassifier

class RealtimeLOBPredictor:
    """
    Real-time LOB mid-price movement predictor.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to trained CatBoost model (.cbm file)
        """
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict mid-price movement.

        Args:
            features: np.ndarray of shape (78,) or (n, 78)

        Returns:
            {
                'prediction': int,        # 0: down, 1: stay, 2: up
                'probabilities': [p0, p1, p2],  # Class probabilities
                'confidence': float       # Max probability
            }
        """
        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = max(probabilities)

        return {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'confidence': float(confidence)
        }
```

#### 6.2.4 전체 통합

**구현 파일**: `realtime_inference/main.py`

```python
import time
from realtime_inference.websocket_client import KISLOBWebSocketClient
from realtime_inference.feature_computer import RealtimeFeatureComputer
from realtime_inference.predictor import RealtimeLOBPredictor

class RealtimeLOBInference:
    """
    End-to-end real-time LOB inference system.
    """

    def __init__(self, model_path: str):
        self.feature_computer = RealtimeFeatureComputer(buffer_size=5)
        self.predictor = RealtimeLOBPredictor(model_path)
        self.ws_client = KISLOBWebSocketClient(
            on_lob_snapshot_callback=self.on_new_lob_snapshot
        )

    def on_new_lob_snapshot(self, lob_snapshot: dict):
        """
        Called when new LOB snapshot is received from WebSocket.

        Pipeline:
        1. Compute 78 features
        2. Run model prediction
        3. Log/display result
        """
        stock_code = lob_snapshot['stock_code']
        timestamp = lob_snapshot['timestamp']

        # 1. Compute features
        features = self.feature_computer.compute_features(lob_snapshot)

        # 2. Predict
        result = self.predictor.predict(features)

        # 3. Display
        prediction_label = ['DOWN ⬇️', 'STAY ➡️', 'UP ⬆️'][result['prediction']]
        confidence = result['confidence']

        print(f"[{timestamp}] {stock_code}: {prediction_label} "
              f"(confidence: {confidence:.2%})")

        # 4. Optional: Trigger actions
        if confidence > 0.80:  # High confidence threshold
            print(f"  🚨 High confidence signal! Consider action.")

    def start(self, app_key: str, app_secret: str, stock_codes: List[str]):
        """Start real-time inference system."""
        print("🚀 Starting real-time LOB inference system...")
        print(f"📊 Monitoring stocks: {stock_codes}")
        print(f"🤖 Model loaded from: {self.predictor.model}")

        # Connect to WebSocket
        self.ws_client.connect(app_key, app_secret, stock_codes)

        # Keep running
        while True:
            time.sleep(1)

# Usage
if __name__ == '__main__':
    inference_system = RealtimeLOBInference(
        model_path='models/catboost_seed_42.cbm'
    )

    inference_system.start(
        app_key='YOUR_KIS_APP_KEY',
        app_secret='YOUR_KIS_APP_SECRET',
        stock_codes=['005930', '000660', ...]  # 10 stocks
    )
```

---

## 7. 성능 목표 및 검증

### 7.1 성능 목표 (PAPER_DRAFT.md 기준)

**Primary Metric: Accuracy**

| Experiment              | Target Accuracy | Std   | p-value  | Seeds |
|-------------------------|----------------|-------|----------|-------|
| Raw baseline (40)       | 68.47%         | 0.39% | -        | 3-5   |
| Raw + Engineered (78)   | **73.43%**     | 0.33% | < 0.001  | 3-5   |
| **Improvement**         | **+4.96 pp**   | -     | -        | -     |

**Secondary Metrics (Per-class)**:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Down  | 0.70      | 0.72   | 0.71     |
| Stay  | 0.66      | 0.65   | 0.65     |
| Up    | 0.76      | 0.75   | 0.75     |

### 7.2 검증 체크리스트

```
✅ Statistical Validation
  - [ ] Train with 3-5 random seeds
  - [ ] Report mean ± std
  - [ ] Run paired t-test vs baseline
  - [ ] Confirm p-value < 0.05
  - [ ] Compute 95% confidence interval

✅ Data Leakage Verification
  - [ ] Temporal split (train < test)
  - [ ] No future information in features
  - [ ] OFI uses Δ(t-1) not Δ(t+1)
  - [ ] Labels not used in feature computation
  - [ ] Normalization fitted on train only

✅ Reproducibility
  - [ ] Fixed random seeds
  - [ ] Pin library versions (requirements.txt)
  - [ ] Save all hyperparameters
  - [ ] Log all experiments
  - [ ] Open-source code

✅ Performance Targets
  - [ ] Accuracy ≥ 73.43% (mean across seeds)
  - [ ] Std < 0.4%
  - [ ] Improvement vs baseline > 4.5 pp
  - [ ] p-value < 0.001
```

---

## 8. 구현 단계

### Phase 1: Feature Engineering (2주)

**목표**: 38개 engineered features 구현 및 검증

**Task 1.1: Raw Feature 추출** (1일)
- [ ] `feature_engineering/raw_features.py` 구현
- [ ] S3 JSONL 파일 → 40 raw features 변환
- [ ] 단위 테스트 작성

**Task 1.2: Engineered Features 구현** (5일)
- [ ] `price_features.py` (6 features) - 1일
- [ ] `volume_features.py` (8 features) - 1일
- [ ] `order_imbalance.py` (6 features) - 1일
- [ ] `order_flow_imbalance.py` (6 features) - 1일
- [ ] `depth_features.py` (6 features) - 0.5일
- [ ] `price_impact.py` (6 features) - 0.5일

**Task 1.3: Pipeline 통합** (2일)
- [ ] `pipeline.py` 구현
- [ ] 78 features 생성 검증
- [ ] Feature 이름 매핑 확인

**Task 1.4: Data Leakage 검증** (2일)
- [ ] `validation/causality_check.py` 구현
- [ ] Temporal causality 검증
- [ ] OFI 계산 로직 검증
- [ ] 모든 체크 통과 확인

**Task 1.5: 성능 최적화** (2일)
- [ ] Numba JIT 적용 (optional)
- [ ] Pre-allocated arrays
- [ ] Batch processing
- [ ] 처리 속도 측정 (target: > 100 snapshots/sec)

**Deliverable**:
- ✅ 78 features 생성 파이프라인
- ✅ 단위 테스트 (coverage > 90%)
- ✅ Data leakage 검증 완료
- ✅ 성능 벤치마크 보고서

---

### Phase 2: Model Training (1주)

**목표**: CatBoost 학습 및 73.43% 목표 달성

**Task 2.1: Label 생성** (1일)
- [ ] k=100 prediction horizon labels
- [ ] 3-class 분류 (down/stay/up)
- [ ] Label distribution 확인

**Task 2.2: Train/Test Split** (0.5일)
- [ ] Temporal split (7 days train, 1 day val, 2 days test)
- [ ] 누수 검증
- [ ] 데이터 분포 확인

**Task 2.3: Single-Seed Baseline** (1일)
- [ ] Raw only (40 features) 학습
- [ ] Raw + Engineered (78 features) 학습
- [ ] 성능 비교

**Task 2.4: Multi-Seed Validation** (2일)
- [ ] 3-5 seeds 학습
- [ ] Mean ± Std 계산
- [ ] Paired t-test
- [ ] 목표 달성 확인 (73.43% ± 0.33%)

**Task 2.5: Hyperparameter Tuning** (1일, optional)
- [ ] Grid search (iterations, depth, lr)
- [ ] 3-fold CV on validation set
- [ ] Best config 선정

**Task 2.6: Model Saving** (0.5일)
- [ ] Save best models (.cbm files)
- [ ] Feature importance 분석
- [ ] 결과 시각화

**Deliverable**:
- ✅ Trained CatBoost models (3-5 seeds)
- ✅ 성능 보고서 (accuracy, p-value, CI)
- ✅ Feature importance ranking
- ✅ Confusion matrix

---

### Phase 3: Real-time Inference (1주)

**목표**: 실시간 추론 시스템 구축

**Task 3.1: WebSocket Client** (2일)
- [ ] `websocket_client.py` 구현
- [ ] KIS API 연동
- [ ] 파이프 구분 형식 파싱

**Task 3.2: Feature Computer** (1일)
- [ ] `feature_computer.py` 구현
- [ ] Real-time buffer 관리
- [ ] 78 features 계산

**Task 3.3: Predictor** (1일)
- [ ] `predictor.py` 구현
- [ ] CatBoost model loading
- [ ] Prediction output formatting

**Task 3.4: 통합 및 테스트** (2일)
- [ ] `main.py` 구현
- [ ] End-to-end 테스트
- [ ] Latency 측정 (target: < 100ms)

**Task 3.5: 배포 및 모니터링** (1일)
- [ ] EC2 배포 (optional)
- [ ] Logging 설정
- [ ] 알림 시스템 (optional)

**Deliverable**:
- ✅ 실시간 추론 시스템
- ✅ Latency < 100ms
- ✅ 장중 실시간 동작 확인

---

### Phase 4: Experiments (선택, 1주)

**목표**: 추가 실험 (PAPER_DRAFT.md 재현)

**Task 4.1: Ablation Study** (2일)
- [ ] Feature group별 기여도 분석
- [ ] Price Impact vs OI vs OFI 비교
- [ ] 결과 보고서

**Task 4.2: Random Baseline** (1일)
- [ ] 38 random features 생성
- [ ] Raw + Random (78) vs Raw + Engineered (78)
- [ ] Domain knowledge 기여도 분리

**Task 4.3: TransLOB Comparison** (2일, optional)
- [ ] TransLOB 구현
- [ ] Raw vs Raw+Engineered 비교
- [ ] CatBoost vs TransLOB

**Task 4.4: Cross-Stock Analysis** (1일)
- [ ] 종목별 성능 분석
- [ ] Feature importance 비교
- [ ] Generalization 평가

**Deliverable**:
- ✅ Ablation study 보고서
- ✅ Random baseline 비교
- ✅ TransLOB 비교 (optional)
- ✅ Cross-stock analysis

---

## 9. 리스크 및 대응

### 9.1 데이터 품질 리스크

**리스크**: S3 데이터에 missing values, outliers 존재 가능

**대응**:
1. 데이터 전처리 시 anomaly detection
2. Missing values forward-fill
3. Outlier filtering (IQR method)

### 9.2 성능 목표 미달 리스크

**리스크**: 73.43% 목표 미달성

**대응**:
1. Hyperparameter tuning (depth, iterations, lr)
2. Feature selection (상위 50개 features만 사용)
3. Ensemble methods (여러 모델 결합)
4. 한국 데이터 특성 반영 (FI-2010과 다를 수 있음)

**Fallback 목표**:
- Minimum acceptable: 70% accuracy (Raw baseline 대비 +1.5 pp)
- 통계적 유의성 (p < 0.05) 유지

### 9.3 실시간 Latency 리스크

**리스크**: Feature 계산 시간 > 100ms

**대응**:
1. Numba JIT 컴파일
2. Pre-computed features (일부)
3. Caching (buffer 재사용)
4. C++ extension (최후 수단)

### 9.4 데이터 누수 리스크

**리스크**: 의도치 않은 look-ahead bias

**대응**:
1. Comprehensive causality check
2. Code review (2명 이상)
3. Temporal split 재검증
4. Feature computation 로직 감사

---

## 10. 참고 자료

### 10.1 핵심 논문

1. **Cont et al. (2014)**: OFI 이론
   "The price impact of order book events"

2. **PAPER_DRAFT.md**: 이 프로젝트의 기준 문서
   - 38 engineered features 정의
   - 73.43% 성능 목표
   - Statistical validation 방법론

3. **Almgren et al. (2005)**: Price impact 이론
   "Direct estimation of equity market impact"

### 10.2 코드 참고

- **FI-2010 Dataset**: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- **CatBoost Documentation**: https://catboost.ai/
- **KIS API Documentation**: (한국투자증권 API 문서)

---

## 요약

**이 명세서는 PAPER_DRAFT.md를 기준으로 작성되었습니다:**

✅ **Raw 40 + Engineered 38 = 78 features**
✅ **CatBoost 학습, 목표 73.43% ± 0.33%**
✅ **Statistical validation (3-5 seeds, p < 0.001)**
✅ **Data leakage 방지 (temporal split, causality check)**
✅ **실시간 추론 시스템 (WebSocket → Features → Prediction)**

**다음 단계**:
1. Phase 1: Feature Engineering 구현 시작
2. S3 데이터 다운로드 및 전처리
3. 38 engineered features 계산 검증
4. Data leakage check 통과 확인

**질문이나 수정 사항이 있으면 언제든지 말씀해주세요!**
