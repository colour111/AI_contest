import pandas as pd
import numpy as np
import os

# 파일 경로 설정 (새로운 v2 폴더)
data_folder = '/Users/namchaewon/Desktop/python/AI_contest/v2/'
os.makedirs(data_folder, exist_ok=True)
file_path = os.path.join(data_folder, 'debt_customer.csv')

# 1. 데이터 불러오기
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 2. 변수 정제 및 결측치 처리
# amount_recovered, outstanding_amount, recovery_time_days의 결측값 처리
df['amount_recovered'] = df['amount_recovered'].fillna(0)
df['outstanding_amount'] = df['original_amount'] - df['amount_recovered']
df['recovery_time_days'] = df.apply(
    lambda row: row['recovery_time_days'] if row['collection_status'] == '성공' else np.nan,
    axis=1
)

# 3. 핵심 변수 생성 (Feature Engineering)
# 회수율: 원금이 0인 경우를 방지
df['collection_rate'] = np.where(df['original_amount'] > 0, df['amount_recovered'] / df['original_amount'], 0)

# 접촉효율성: 접촉 시도 횟수가 0인 경우를 방지
df['contact_efficiency'] = np.where(df['contact_attempts'] > 0, df['amount_recovered'] / df['contact_attempts'], 0)

# 4. 최종 데이터 확인
print("\n[전처리 후 데이터프레임]")
print(df.head())
print("\n[전처리 후 데이터 정보]")
print(df.info())

# 전처리된 데이터 임시 저장
preprocessed_file_path = os.path.join(data_folder, 'preprocessed_debt_customer.csv')
df.to_csv(preprocessed_file_path, index=False, encoding='utf-8-sig')
print(f"\n전처리된 데이터가 성공적으로 저장되었습니다: {preprocessed_file_path}")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

# 파일 경로 설정 (새로운 v2 폴더)
data_folder = '/Users/namchaewon/Desktop/python/AI_contest/v2/'
preprocessed_file_path = os.path.join(data_folder, 'preprocessed_debt_customer.csv')

# 1. 전처리된 데이터 불러오기
try:
    df = pd.read_csv(preprocessed_file_path, encoding='utf-8-sig')
    print("전처리된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 전 단계에서 생성한 파일명을 확인해주세요.")
    exit()

# 2. 군집 분석에 사용할 변수 선택 및 스케일링
features = ['age', 'original_amount', 'days_past_due', 'contact_attempts', 
            'credit_score', 'income_level', 'collection_rate']
df_clustering = df[features].copy()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)
print("\n[군집 분석을 위한 데이터 표준화 완료]")

# 3. 계층적 군집 분석 및 덴드로그램 생성
linked = linkage(df_scaled, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

print("\n위 덴드로그램을 참고하여 비즈니스 목적에 맞는 군집 수를 결정하세요.")
print("덴드로그램에서 y축(거리)의 특정 지점을 수평선으로 자른다고 생각했을 때,")
print("수평선 아래에 생성되는 세로선 그룹의 개수가 군집의 수가 됩니다.")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 파일 경로 설정 (새로운 v2 폴더)
data_folder = '/Users/namchaewon/Desktop/python/AI_contest/v2/'
preprocessed_file_path = os.path.join(data_folder, 'preprocessed_debt_customer.csv')

# 1. 전처리된 데이터 불러오기
try:
    df = pd.read_csv(preprocessed_file_path, encoding='utf-8-sig')
    print("전처리된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 2. 군집 분석에 사용할 변수 선택 및 스케일링
features = ['age', 'original_amount', 'days_past_due', 'contact_attempts', 
            'credit_score', 'income_level', 'collection_rate']
df_clustering = df[features].copy()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# 3. K-Means 모델 학습 (K=3으로 가정)
optimal_k = 3
print(f"\n최적의 K={optimal_k}로 군집 분석을 진행합니다.")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(df_scaled)

# 4. 군집별 주요 특성 분석 및 이름, 전략 정의
cluster_summary = df.groupby('cluster').agg(
    avg_original_amount=('original_amount', 'mean'),
    avg_days_past_due=('days_past_due', 'mean'),
    avg_collection_rate=('collection_rate', 'mean'),
    avg_contact_attempts=('contact_attempts', 'mean'),
    count=('debtor_id', 'count')
).round(2)
print("\n[군집별 주요 특성 분석 결과]")
print(cluster_summary)

# 분석 결과를 바탕으로 군집에 의미있는 이름과 전략 매핑
cluster_names = {
    0: '저위험_단기연체군',
    1: '고액_고위험군',
    2: '중위험_장기연체군'
}
strategy_map = {
    '저위험_단기연체군': '자동 문자/이메일 발송',
    '고액_고위험군': '전담 추심원 배정 및 법적 절차 준비',
    '중위험_장기연체군': '채무 조정(분할 상환, 유예) 제안'
}
df['cluster_name'] = df['cluster'].map(cluster_names)
df['recommended_strategy'] = df['cluster_name'].map(strategy_map)

# 5. 최종 데이터셋 저장
final_file_path = os.path.join(data_folder, 'debt_customer_analysis.csv')
df.to_csv(final_file_path, index=False, encoding='utf-8-sig')
print(f"\n군집 정보 및 전략이 포함된 최종 데이터가 '{final_file_path}' 파일로 저장되었습니다.")
