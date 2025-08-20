import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Faker 라이브러리를 한국어로 설정
fake = Faker('ko_KR')

# 더미 데이터 생성
num_data = 2000
data = {
    'debtor_id': range(1, num_data + 1),
    'age': np.random.randint(20, 70, size=num_data),
    'gender': np.random.choice(['남성', '여성'], size=num_data),
    'region': [fake.city() for _ in range(num_data)],
    'original_amount': np.random.randint(100000, 50000000, size=num_data),
    'days_past_due': np.random.randint(30, 365, size=num_data) # 30일 이상으로만 생성
}

df = pd.DataFrame(data)

# 연체 기간 그룹화
# ~1개월 : 초단기 (제외), 2~3개월 : 단기, 4~6개월 : 중기, 6개월~ : 상각
def get_delinquency_group(days):
    if 30 <= days < 90:
        return '단기'
    elif 90 <= days < 180:
        return '중기'
    else:
        return '상각'
    
df['delinquency_group'] = df['days_past_due'].apply(get_delinquency_group)

# 연체 기간 그룹별 회수율 설정 (요청된 비율 적용)
collection_rates = {
    '단기': 0.12,  # 12%
    '중기': 0.035, # 3.5% (3~4% 내외)
    '상각': 0.0035 # 0.35%
}

# 요청된 회수율을 기반으로 collection_status 및 amount_recovered 생성
def calculate_collection_status_and_amount(row):
    rate = collection_rates.get(row['delinquency_group'], 0)
    
    # 설정된 회수율에 따라 성공/실패 여부 결정
    if random.random() < rate:
        collection_status = '성공'
        # 성공 시, 원금의 10% ~ 100% 사이의 금액으로 회수금액 설정
        amount_recovered = np.random.randint(int(row['original_amount'] * 0.1), int(row['original_amount']))
    else:
        collection_status = '실패'
        amount_recovered = 0
        
    return collection_status, amount_recovered

df[['collection_status', 'amount_recovered']] = df.apply(
    lambda row: pd.Series(calculate_collection_status_and_amount(row)),
    axis=1
)

df['outstanding_amount'] = df['original_amount'] - df['amount_recovered']

# 나머지 변수들 생성 (이전과 동일)
df['strategy_used'] = df.apply(
    lambda row: np.random.choice(['전화', '방문'], p=[0.7, 0.3]) if row['days_past_due'] > 90 else np.random.choice(['문자', '전화'], p=[0.8, 0.2]),
    axis=1
)
df['contact_attempts'] = df['days_past_due'].apply(
    lambda x: int(np.random.normal(x / 30, 1)) + 1
)
df['contact_success_count'] = df['contact_attempts'].apply(
    lambda x: int(np.random.beta(a=0.5, b=5) * x)
)
df['recovery_time_days'] = df.apply(
    lambda row: np.random.randint(1, row['days_past_due']) if row['collection_status'] == '성공' else np.nan,
    axis=1
)

df['credit_score'] = np.random.randint(400, 950, size=num_data)
df['income_level'] = np.random.randint(2000, 8000, size=num_data)
df['employment_status'] = np.random.choice(['직장인', '자영업', '무직'], size=num_data, p=[0.7, 0.2, 0.1])
df['loan_id'] = [f'L{i:04d}' for i in range(1, num_data + 1)]

# 불필요한 값 조정 및 데이터 형식 변경
df['contact_attempts'] = df['contact_attempts'].clip(lower=1)
df['contact_success_count'] = df['contact_success_count'].clip(lower=0)

# 최종 데이터 확인
print(df.head())
print(df.info())

# 파일 경로 및 이름 설정
file_path = '/Users/namchaewon/Desktop/python/AI_contest/debt_customer.csv'

# 디렉토리가 존재하지 않으면 생성
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# CSV 파일로 저장 (인코딩을 'utf-8-sig'로 지정하여 한글 깨짐 방지)
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"\n현실적 회수율을 반영한 데이터가 성공적으로 다음 경로에 저장되었습니다: {file_path}")