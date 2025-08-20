import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# 페이지 기본 설정
st.set_page_config(
    page_title="채무자 예측 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터와 모델을 한 번만 로드하는 함수 (캐싱)
@st.cache_data
def load_data_and_model(file_path):
    """
    데이터를 불러오고, 모델을 학습시키며, 스케일러를 반환합니다.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 모델 학습에 필요한 변수 선택 및 스케일링
        # original_amount, days_past_due, credit_score, collection_rate는 예측에 주요 변수
        features = ['age', 'original_amount', 'days_past_due', 'contact_attempts', 
                    'credit_score', 'income_level', 'collection_rate']
        
        # 가상의 더미데이터이므로, 실제 로직에서는 NaN 값 처리 필요
        df = df.dropna(subset=features)
        df_clustering = df[features].copy()
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clustering)
        
        # K-Means 모델 학습 (K=3으로 고정)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        kmeans.fit(df_scaled)
        
        return df, scaler, kmeans
    
    except FileNotFoundError:
        st.error(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return None, None, None

# 파일 경로 설정 (v2 폴더)
data_file_path = '/Users/namchaewon/Desktop/python/AI_contest/v2/debt_customer_analysis.csv'
df, scaler, kmeans = load_data_and_model(data_file_path)

# 군집 이름 및 전략 매핑
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

# ----------------- UI 구성 시작 -----------------
st.title('💰 신규 채무자 분석 및 예측 대시보드')
st.markdown('### 고객 정보를 입력하고 맞춤형 회수 전략 및 성공 확률을 확인하세요.')

if df is not None:
    # 사용자 입력 섹션
    st.header('1. 고객 정보 입력')
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        age = st.slider('나이(Age)', 20, 70, 35)
        original_amount = st.number_input('채무 금액(원)', min_value=100000, max_value=50000000, value=15000000)
        days_past_due = st.number_input('연체일(Days Past Due)', min_value=30, max_value=365, value=90)
    
    with col_input2:
        gender = st.selectbox('성별(Gender)', options=['남성', '여성'])
        employment_status = st.selectbox('직업(Employment Status)', options=['직장인', '자영업', '무직'])
        credit_score = st.slider('신용점수(Credit Score)', 400, 950, 700)
        
    st.markdown('---')

    # 예측 섹션
    st.header('2. 분석 및 예측 결과')
    
    # 2-1. 입력값을 기반으로 회수 성공 확률 계산
    # 연체 기간 그룹에 따라 회수율을 직접 적용
    if 30 <= days_past_due < 90:
        rate = 0.12 # 단기 (12%)
    elif 90 <= days_past_due < 180:
        rate = 0.035 # 중기 (3.5%)
    else:
        rate = 0.0035 # 상각 (0.35%)

    # 신용점수와 채무금액에 따라 확률 조정
    # 신용점수 700점을 기준으로 보정
    credit_adjustment = (credit_score - 700) / 500
    # 금액이 클수록 확률이 감소
    amount_adjustment = (15000000 - original_amount) / 50000000 * 0.5

    collection_success_prob = (rate * 100) + (credit_adjustment * 5) + (amount_adjustment * 10)
    collection_success_prob = max(0, min(100, collection_success_prob))
    
    st.subheader('회수 성공 확률')
    prob_col = st.columns(1)[0]
    prob_col.metric(label="예상 회수 성공률", value=f"{collection_success_prob:.2f}%")
    st.progress(collection_success_prob / 100)
    st.write(f"예측된 성공률은 {collection_success_prob:.2f}% 입니다. ")
    
    st.markdown('---')

    # 2-2. 입력값을 기반으로 군집 예측 및 전략 제시
    # 예측에 필요한 변수 (원본 데이터셋의 평균값 사용)
    contact_attempts_avg = df['contact_attempts'].mean()
    income_level_avg = df['income_level'].mean()
    collection_rate_avg = df['collection_rate'].mean()

    input_data = pd.DataFrame([{
        'age': age,
        'original_amount': original_amount,
        'days_past_due': days_past_due,
        'contact_attempts': contact_attempts_avg,
        'credit_score': credit_score,
        'income_level': income_level_avg,
        'collection_rate': collection_rate_avg
    }])

    # 입력 데이터 스케일링
    input_scaled = scaler.transform(input_data)
    
    # 군집 예측
    predicted_cluster = kmeans.predict(input_scaled)[0]
    
    # 예측된 군집에 따른 이름과 전략
    predicted_cluster_name = cluster_names.get(predicted_cluster, "알 수 없음")
    recommended_strategy = strategy_map.get(predicted_cluster_name, "알 수 없음")
    
    st.subheader('예측 결과 및 추천 전략')
    
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric(label="예측된 군집", value=predicted_cluster_name)
    with result_col2:
        st.metric(label="추천 회수 전략", value=recommended_strategy)
    
    st.info(f"이 고객은 **'{predicted_cluster_name}'** 군집에 속할 가능성이 높으며, **'{recommended_strategy}'** 전략이 가장 효과적일 것으로 예상됩니다.")
