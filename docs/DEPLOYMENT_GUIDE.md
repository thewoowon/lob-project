# 🚀 AWS 배포 가이드: 한국 주식 LOB 데이터 수집

## 📋 목차
1. [사전 준비](#사전-준비)
2. [AWS 인프라 구축](#aws-인프라-구축)
3. [키움 API 설정](#키움-api-설정)
4. [데이터 수집 시작](#데이터-수집-시작)
5. [모니터링 및 관리](#모니터링-및-관리)
6. [비용 관리](#비용-관리)

---

## 사전 준비

### 1. 필수 계정 및 권한

✅ **AWS 계정**
- 신용카드 등록 필요
- IAM 사용자 생성 (Administrator 권한)

✅ **키움증권 계정**
- 증권 계좌 개설
- OpenAPI 신청: [키움증권 OpenAPI](https://www3.kiwoom.com/nkw.templateFrameSet.do?m=m1408000000)
- 승인까지 1-2 영업일 소요

✅ **개발 환경**
- Terraform 설치: https://www.terraform.io/downloads
- AWS CLI 설치: https://aws.amazon.com/cli/
- Python 3.11+

### 2. AWS CLI 설정

```bash
# AWS CLI 설정
aws configure

# 입력 정보:
# AWS Access Key ID: (IAM 사용자 액세스 키)
# AWS Secret Access Key: (IAM 시크릿 키)
# Default region: ap-northeast-2 (서울)
# Default output format: json
```

### 3. 프로젝트 클론

```bash
git clone https://github.com/yourusername/lob-project.git
cd lob-project
```

---

## AWS 인프라 구축

### Step 1: Terraform 변수 설정

```bash
cd aws_setup

# variables.tf 생성
cat > variables.tfvars <<EOF
aws_region       = "ap-northeast-2"
project_name     = "lob-collector"
s3_bucket_name   = "your-unique-bucket-name"  # 변경 필요!
db_password      = "YourSecurePassword123!"    # 변경 필요!
alert_email      = "your-email@example.com"    # 변경 필요!
EOF
```

### Step 2: Terraform 초기화 및 계획

```bash
# Terraform 초기화
terraform init

# 계획 확인 (생성될 리소스 검토)
terraform plan -var-file=variables.tfvars

# 예상 비용 확인
# - EC2 t3.medium (Windows): ~$35/월
# - RDS t3.micro (PostgreSQL): ~$15/월
# - S3 storage: ~$1/월
# - 총 예상: ~$50/월
```

### Step 3: 인프라 생성

```bash
# 인프라 배포 (약 10-15분 소요)
terraform apply -var-file=variables.tfvars

# 출력 정보 저장
terraform output > outputs.txt

# 중요 정보 확인:
# - EC2 Public IP
# - S3 Bucket Name
# - RDS Endpoint
```

**생성되는 리소스:**
- ✅ VPC + 서브넷 (Public/Private)
- ✅ EC2 Windows Server 2022 (t3.medium)
- ✅ RDS PostgreSQL 15 (db.t3.micro)
- ✅ S3 버킷 (버저닝 + 라이프사이클)
- ✅ Security Groups
- ✅ IAM Roles
- ✅ CloudWatch Alarms

---

## 키움 API 설정

### Step 1: EC2 접속

```bash
# EC2 Public IP 확인
EC2_IP=$(terraform output -raw ec2_public_ip)

# Windows RDP 접속
# - Windows: mstsc.exe 실행 → IP 입력
# - Mac: Microsoft Remote Desktop 앱 사용
# - Linux: rdesktop 또는 freerdp 사용

# 접속 정보:
# - Host: <EC2_IP>
# - Username: Administrator
# - Password: (AWS Console에서 Get Password 클릭)
```

### Step 2: 키움 OpenAPI 설치

EC2 Windows 내부에서:

1. **키움 OpenAPI 설치**
   - 브라우저 열기
   - https://www3.kiwoom.com/nkw.templateFrameSet.do?m=m1408000000 접속
   - "OpenAPI+ 다운로드" 클릭
   - 설치 실행

2. **KOA Studio 실행**
   - 바탕화면의 KOA Studio 실행
   - 키움 계정으로 로그인
   - 정상 로그인 확인

3. **자동 로그인 설정** (Optional)
   ```
   KOA Studio → 설정 → 자동 로그인 체크
   ```

### Step 3: 프로젝트 코드 복사

EC2 Windows에서 PowerShell 실행:

```powershell
# 프로젝트 클론 (이미 user_data에서 실행됨)
cd C:\lob-project

# 수집기 설정 수정
notepad lob_preprocessing\data\kiwoom_collector.py

# S3 버킷 이름 수정 (lines 주석 참조)
```

---

## 데이터 수집 시작

### 방법 1: 수동 실행 (테스트용)

EC2 Windows에서:

```powershell
# PowerShell 관리자 권한으로 실행
cd C:\lob-project\lob_preprocessing

# 수집 시작 (삼성전자 + 크래프톤)
python data\kiwoom_collector.py `
  --codes 005930 259960 `
  --s3-bucket YOUR_BUCKET_NAME

# 장중 자동 수집 모드
python data\kiwoom_collector.py `
  --codes 005930 259960 `
  --s3-bucket YOUR_BUCKET_NAME `
  --auto
```

### 방법 2: 자동 실행 (Task Scheduler)

Task Scheduler가 이미 설정되어 있음 (user_data):
- 매일 08:30 AM 자동 실행
- 장 시작(9:00) 전 대기
- 장 종료(15:30) 후 자동 종료

**Task 확인:**
```powershell
# Task Scheduler 열기
taskschd.msc

# "LOBCollector" 태스크 확인
# - Trigger: Daily 8:30 AM
# - Action: C:\start_collector.bat
```

**수동 트리거:**
```powershell
# 즉시 실행
schtasks /Run /TN "LOBCollector"
```

---

## 모니터링 및 관리

### 1. 데이터 확인

#### 로컬 파일 확인 (EC2)
```powershell
# 로컬 저장 경로
cd C:\lob-data

# 날짜별 폴더 확인
dir

# 최신 파일 확인
dir 20240101\*.parquet | sort LastWriteTime -Descending | select -First 5
```

#### S3 확인
```bash
# AWS CLI로 S3 확인
aws s3 ls s3://YOUR_BUCKET_NAME/kospi/005930/

# 특정 날짜 데이터 다운로드
aws s3 cp s3://YOUR_BUCKET_NAME/kospi/005930/20240101/ . --recursive
```

### 2. 로그 확인

#### 수집 로그
```powershell
# EC2에서 로그 확인
notepad C:\lob-data\logs\collector_20240101.log

# 실시간 로그 tail (PowerShell)
Get-Content C:\lob-data\logs\collector_20240101.log -Wait
```

#### CloudWatch Logs (설정 시)
```bash
# AWS CLI로 로그 확인
aws logs tail /aws/ec2/kiwoom-collector --follow
```

### 3. CloudWatch Alarms

자동 설정된 알람:
- ✅ **EC2 CPU > 80%**: 이메일 알림
- ✅ **RDS Storage < 10%**: 이메일 알림
- ✅ **데이터 수집 중단**: (Custom metric 필요)

### 4. 데이터 품질 체크

```python
# 로컬에서 데이터 품질 확인
import pandas as pd

# S3에서 다운로드한 데이터 로드
df = pd.read_parquet('005930_20240101_153000.parquet')

print(f"Records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum()}")

# Mid-price 계산 확인
df['mid_price'] = (df['매수호가1'] + df['매도호가1']) / 2
print(df['mid_price'].describe())
```

---

## 비용 관리

### 월간 예상 비용 (~$50)

| 항목 | 사양 | 비용 |
|------|------|------|
| EC2 Windows | t3.medium (2vCPU, 4GB) | ~$35/월 |
| RDS PostgreSQL | db.t3.micro (1vCPU, 1GB) | ~$15/월 |
| S3 Storage | ~10GB/월 | ~$0.23/월 |
| Data Transfer | ~1GB/월 | ~$0.10/월 |
| **총계** | | **~$50/월** |

### 비용 절감 팁

1. **EC2 인스턴스 예약**
   ```
   1년 예약 → 30% 할인
   3년 예약 → 50% 할인
   ```

2. **장 종료 후 인스턴스 중지**
   ```powershell
   # Task Scheduler: 매일 16:00에 중지
   # Task Scheduler: 매일 08:30에 시작
   → 월 비용 ~60% 절감
   ```

3. **S3 Lifecycle 정책** (이미 설정됨)
   ```
   30일 후 → Glacier ($0.004/GB/월)
   90일 후 → Deep Archive ($0.00099/GB/월)
   ```

4. **RDS 스냅샷 정리**
   ```bash
   # 오래된 스냅샷 삭제
   aws rds delete-db-snapshot --db-snapshot-identifier old-snapshot
   ```

### 비용 알림 설정

```bash
# AWS Budgets 설정
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget file://budget.json

# budget.json:
{
  "BudgetName": "LOB-Monthly-Budget",
  "BudgetLimit": {
    "Amount": "60",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

---

## 트러블슈팅

### 문제 1: 키움 API 로그인 실패

**증상:** "로그인 오류" 메시지

**해결:**
1. KOA Studio에서 수동 로그인 테스트
2. OpenAPI 신청 승인 확인
3. 계좌 비밀번호 확인

### 문제 2: S3 업로드 실패

**증상:** "Access Denied" 에러

**해결:**
```bash
# IAM Role 확인
aws iam get-instance-profile --instance-profile-name lob-collector-ec2-profile

# S3 권한 테스트
aws s3 ls s3://YOUR_BUCKET_NAME/

# 권한 없으면 Terraform 재적용
terraform apply -var-file=variables.tfvars
```

### 문제 3: 데이터 수집 중단

**증상:** 로그에 "Connection lost"

**해결:**
1. 키움 API 재로그인
2. 프로세스 재시작:
   ```powershell
   taskkill /IM python.exe /F
   python data\kiwoom_collector.py --auto
   ```

### 문제 4: EC2 비용 초과

**증상:** 월 $50 초과

**해결:**
1. 인스턴스 타입 다운그레이드:
   ```hcl
   # terraform_main.tf 수정
   instance_type = "t3.small"  # t3.medium → t3.small
   ```
2. 장 종료 후 인스턴스 중지:
   ```bash
   aws ec2 stop-instances --instance-ids i-xxxxx
   ```

---

## 정리 (Cleanup)

### 전체 인프라 삭제

```bash
cd aws_setup

# 주의: 모든 데이터가 삭제됩니다!
terraform destroy -var-file=variables.tfvars

# S3 버킷 수동 비우기 (필요 시)
aws s3 rm s3://YOUR_BUCKET_NAME --recursive
```

---

## 다음 단계

1. ✅ 데이터 수집 (2-4주)
2. ✅ 데이터 품질 확인
3. ✅ 전처리 파이프라인 실행
4. ✅ 크립토 vs 한국 주식 비교 실험
5. ✅ 논문 작성!

---

## 연락 및 지원

- 문제 발생 시: GitHub Issues
- AWS 관련: AWS Support
- 키움 API: 키움증권 고객센터 (1544-9000)

**Good luck! 🚀**
