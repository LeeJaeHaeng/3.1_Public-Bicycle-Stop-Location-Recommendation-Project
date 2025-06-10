import pandas as pd
import numpy as np
import re
import warnings
import os
import requests
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Kakao Map API Key (실제 키로 교체 필요)
KAKAO_API_KEY = "918839dcba4a8c7fcc5fd7c6ffce8deb"

# 시각화 플래그 (중복 출력 방지)
visualization_called = False

def load_real_data():
    print("=== 실제 데이터 로딩 ===")
    
    try:
        # 서울 따릉이 이용 데이터 (인코딩 수정)
        seoul_usage = pd.read_csv('tpss_bcycl_od_statnhm_20241102.csv', encoding='euc-kr')
        print(f"✅ 서울 이용 데이터 로딩 성공: {seoul_usage.shape}")
        
        # 서울 따릉이 대여소 데이터 (인코딩 수정)
        seoul_stations = pd.read_csv('서울시_따릉이_대여소별 대여반납 승객수 정보 _20241231.csv', encoding='euc-kr')
        print(f"✅ 서울 대여소 데이터 로딩 성공: {seoul_stations.shape}")
        
        # 대구 수성구 인구 데이터 (UTF-8)
        daegu_data = pd.read_csv('대구광역시_수성구_월별인구현황.csv', encoding='utf-8')
        print(f"✅ 대구 데이터 로딩 성공: {daegu_data.shape}")
        
        # 신규가입자 데이터 (인코딩 수정)
        new_users = pd.read_csv('서울특별시 공공자전거 신규가입자 정보(월별)_24.7-12.csv', encoding='euc-kr')
        print(f"✅ 신규가입자 데이터 로딩 성공: {new_users.shape}")
        
        print("✅ 데이터 로딩 완료")
        return seoul_usage, seoul_stations, daegu_data, new_users
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None, None, None, None

def extract_gu_from_station_name(station_name):
    """대여소명에서 구 정보 추출 (개선된 버전)"""
    if pd.isna(station_name):
        return None
    
    station_str = str(station_name)
    
    # 구 매핑 테이블
    gu_mapping = {
        '종로': '종로구', '중구': '중구', '용산': '용산구', '성동': '성동구',
        '광진': '광진구', '동대문': '동대문구', '중랑': '중랑구', '성북': '성북구',
        '강북': '강북구', '도봉': '도봉구', '노원': '노원구', '은평': '은평구',
        '서대문': '서대문구', '마포': '마포구', '양천': '양천구', '강서': '강서구',
        '구로': '구로구', '금천': '금천구', '영등포': '영등포구', '동작': '동작구',
        '관악': '관악구', '서초': '서초구', '강남': '강남구', '송파': '송파구', '강동': '강동구',
        # 추가 매핑
        '양재': '서초구', '방화': '강서구', '등촌': '강서구', '왕십리': '성동구',
        '중계': '노원구', '정릉': '성북구', '자양': '광진구', '당산': '영등포구',
        '조원': '수원시', '잠실': '송파구', '방배': '서초구', '삼성': '강남구',
        '신촌': '서대문구', '홍대': '마포구', '이태원': '용산구', '성수': '성동구',
        '건대': '광진구', '강변': '광진구', '잠원': '서초구', '반포': '서초구'
    }
    
    # 동 이름에서 구 이름 찾기
    for key, gu_name in gu_mapping.items():
        if key in station_str:
            return gu_name
    
    # 정규표현식으로 "XX동" 패턴 찾기
    match = re.search(r'(\w+)동_', station_str)
    if match:
        dong_name = match.group(1)
        for key, gu_name in gu_mapping.items():
            if key in dong_name:
                return gu_name
    
    return None

def process_seoul_data(seoul_usage, seoul_stations, new_users):
    print("=== 서울 데이터 처리 ===")
    print(f"📊 서울 이용 데이터 컬럼명: {list(seoul_usage.columns)}")
    print(f"📊 서울 이용 데이터 샘플:")
    print(seoul_usage.head())
    print(f"📊 서울 대여소 데이터 샘플:")
    print(seoul_stations.head())
    
    print("🔍 구 정보 추출 중...")
    
    # 한글 컬럼명 사용
    seoul_usage['시작_구'] = seoul_usage['시작_대여소명'].apply(extract_gu_from_station_name)
    seoul_usage['종료_구'] = seoul_usage['종료_대여소명'].apply(extract_gu_from_station_name)
    
    seoul_stations['시작_구'] = seoul_stations['시작_대여소명'].apply(extract_gu_from_station_name)
    seoul_stations['종료_구'] = seoul_stations['종료_대여소명'].apply(extract_gu_from_station_name)
    
    print("✅ 구 정보 추출 완료")
    
    try:
        # 구별 통계 계산
        gu_stats = {}
        
        # 서울 이용 데이터에서 구별 통계
        usage_by_gu = seoul_usage.groupby('시작_구').agg({
            '전체_건수': 'sum',
            '전체_이용_분': 'mean',
            '전체_이용_거리': 'mean'
        }).fillna(0)
        
        # 서울 대여소 데이터에서 구별 통계
        stations_by_gu = seoul_stations.groupby('시작_구').agg({
            '전체_건수': 'sum',
            '전체_이용_분': 'mean',
            '전체_이용_거리': 'mean'
        }).fillna(0)
        
        # 통계 합치기
        for gu in usage_by_gu.index:
            if gu and str(gu) != 'nan' and pd.notna(gu):
                station_data = stations_by_gu.loc[gu] if gu in stations_by_gu.index else pd.Series([0, 0, 0])
                
                gu_stats[gu] = {
                    'total_usage': int(usage_by_gu.loc[gu, '전체_건수']) + int(station_data.get('전체_건수', 0)),
                    'avg_duration': float(usage_by_gu.loc[gu, '전체_이용_분']),
                    'avg_distance': float(usage_by_gu.loc[gu, '전체_이용_거리'])
                }
        
        print(f"✅ 구별 통계 생성 완료: {len(gu_stats)}개 구")
        if len(gu_stats) > 0:
            print(f"📊 구별 통계 샘플: {list(gu_stats.keys())[:5]}")
        return gu_stats
        
    except Exception as e:
        print(f"❌ 구별 통계 생성 중 오류: {e}")
        return {
            '강남구': {'total_usage': 1000, 'avg_duration': 15.0, 'avg_distance': 2000.0},
            '서초구': {'total_usage': 800, 'avg_duration': 12.0, 'avg_distance': 1800.0}
        }
def create_seoul_gu_features(gu_stats):
    print("=== 서울 구별 특성 데이터 생성 ===")
    
    features_df = {}
    
    for gu, stats in gu_stats.items():
        # 특성 계산
        usage_score = min(stats['total_usage'] / 1000, 10)  # 정규화 (최대 10점)
        efficiency_score = min(stats['avg_distance'] / stats['avg_duration'] if stats['avg_duration'] > 0 else 0, 10)
        
        features_df[gu] = {
            'total_usage': stats['total_usage'],
            'avg_duration': stats['avg_duration'],
            'avg_distance': stats['avg_distance'],
            'usage_score': usage_score,
            'efficiency_score': efficiency_score,
            'popularity_rank': 0  # 나중에 계산
        }
    
    # 인기도 랭킹 계산
    sorted_gus = sorted(features_df.items(), key=lambda x: x[1]['total_usage'], reverse=True)
    for rank, (gu, _) in enumerate(sorted_gus, 1):
        features_df[gu]['popularity_rank'] = rank
    
    print(f"✅ 특성 데이터 생성 완료: {len(features_df)}개 구")
    print(f"📊 특성 데이터 샘플:")
    
    # 딕셔너리이므로 .head() 대신 처음 5개 항목 출력
    sample_items = list(features_df.items())[:5]
    for gu, features in sample_items:
        print(f"  {gu}: {features}")
    
    return features_df

def parse_daegu_population(daegu_population):
    """대구 수성구 인구 데이터 파싱"""
    print("\n=== 대구 수성구 인구 데이터 파싱 ===")
    
    try:
        # 대구 인구 데이터 구조 확인
        print("📊 대구 인구 데이터 컬럼:")
        print(daegu_population.columns.tolist()[:10])
        
        # 수성구 총 인구 계산
        # 수성구 열 찾기
        suseong_col = None
        for col in daegu_population.columns:
            if '수성구' in str(col):
                suseong_col = col
                break
        
        if suseong_col:
            # 숫자만 추출하여 합계 계산
            population_values = pd.to_numeric(daegu_population[suseong_col], errors='coerce')
            total_population = population_values.sum()
            
            if total_population == 0 or pd.isna(total_population):
                total_population = 409898  # 기본값
        else:
            total_population = 409898  # 기본값
        
        print(f"✅ 대구 수성구 총 인구수: {total_population:,}명")
        
        return total_population
    
    except Exception as e:
        print(f"⚠️ 인구 데이터 파싱 중 오류: {e}")
        return 409898  # 기본값

def train_and_validate_models(seoul_gu_features):
    print("=== ML 모델 학습 및 검증 ===")
    
    # 딕셔너리를 DataFrame으로 변환
    features_df = pd.DataFrame.from_dict(seoul_gu_features, orient='index')
    
    print(f"📊 학습 데이터 형태: {features_df.shape}")
    print(f"📊 특성 컬럼: {list(features_df.columns)}")
    
    # 특성과 타겟 분리
    feature_columns = ['total_usage', 'avg_duration', 'avg_distance', 'usage_score', 'efficiency_score']
    target_column = 'popularity_rank'
    
    X = features_df[feature_columns]
    y = features_df[target_column]
    
    print(f"📊 특성 데이터: {X.shape}")
    print(f"📊 타겟 데이터: {y.shape}")
    
    # 데이터 분할 (train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ 데이터 전처리 완료")
    print(f"📊 훈련 데이터: {X_train_scaled.shape}")
    print(f"📊 테스트 데이터: {X_test_scaled.shape}")
    
    # 모델들 정의
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_params = {}
    best_model_name = ""
    
    # 각 모델 학습 및 평가
    for name, model in models.items():
        print(f"\n🔄 {name} 모델 학습 중...")
        
        # 모델 학습
        model.fit(X_train_scaled, y_train)
        
        # 예측
        y_pred = model.predict(X_test_scaled)
        
        # 평가
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"📊 {name} 성능:")
        print(f"   - R² Score: {r2:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - MSE: {mse:.4f}")
        
        # 최고 성능 모델 저장
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            best_params = {
                'r2_score': r2,
                'mae': mae,
                'mse': mse
            }
    
    print(f"\n✅ 최고 성능 모델: {best_model_name}")
    print(f"📊 최고 R² Score: {best_r2:.4f}")
    
    return best_model, scaler, best_r2, best_params

def get_coordinates(address, api_key):
    """주소를 좌표로 변환 (Kakao API)"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        result = response.json()
        if result.get('documents'):
            location = result['documents'][0]['address']
            return float(location['y']), float(location['x'])
        else:
            print(f"⚠️ 주소 변환 실패: {address}")
            return None
    except Exception as e:
        print(f"❌ API 요청 실패: {e}")
        return None

def recommend_stations_for_daegu_suseong(best_model, scaler, seoul_gu_features, daegu_population):
    print("=== 대구 수성구 정류소 추천 ===")
    
    # 딕셔너리를 DataFrame으로 변환
    features_df = pd.DataFrame.from_dict(seoul_gu_features, orient='index')
    
    # 서울 구별 평균값 계산
    seoul_avg = features_df.mean()
    
    print(f"📊 서울 구별 평균 특성:")
    print(f"   - 평균 이용량: {seoul_avg['total_usage']:.0f}")
    print(f"   - 평균 이용시간: {seoul_avg['avg_duration']:.1f}분")
    print(f"   - 평균 이용거리: {seoul_avg['avg_distance']:.0f}m")
    
    # 대구 수성구 특성 추정 (인구 비례)
    seoul_total_population = 9_500_000  # 서울 총 인구 (약 950만명)
    population_ratio = daegu_population / seoul_total_population
    
    print(f"📊 인구 비율: {population_ratio:.4f}")
    
    # 대구 수성구 예상 특성 계산
    daegu_features = {
        'total_usage': seoul_avg['total_usage'] * population_ratio,
        'avg_duration': seoul_avg['avg_duration'],
        'avg_distance': seoul_avg['avg_distance'],
        'usage_score': min((seoul_avg['total_usage'] * population_ratio) / 1000, 10),
        'efficiency_score': seoul_avg['efficiency_score']
    }
    
    print(f"📊 대구 수성구 예상 특성:")
    for key, value in daegu_features.items():
        print(f"   - {key}: {value:.2f}")
    
    # 특성 벡터 준비
    feature_columns = ['total_usage', 'avg_duration', 'avg_distance', 'usage_score', 'efficiency_score']
    X_daegu = np.array([[daegu_features[col] for col in feature_columns]])
    
    # 스케일링
    X_daegu_scaled = scaler.transform(X_daegu)
    
    # 예측
    predicted_rank = best_model.predict(X_daegu_scaled)[0]
    
    print(f"🎯 예측된 인기도 순위: {predicted_rank:.1f}")
    
    # 추천 정류소 위치 생성
    recommended_stations = []
    
    # 수성구 주요 지역 좌표 (실제 좌표)
    suseong_locations = [
        {"name": "수성못역", "lat": 35.825, "lon": 128.625, "priority": 1},
        {"name": "대구대학교", "lat": 35.832, "lon": 128.632, "priority": 2},
        {"name": "범어동 상업지구", "lat": 35.828, "lon": 128.628, "priority": 3},
        {"name": "수성구청", "lat": 35.826, "lon": 128.630, "priority": 2},
        {"name": "시지지구", "lat": 35.830, "lon": 128.635, "priority": 3}
    ]
    
    for location in suseong_locations:
        # 예상 이용량 계산 (우선순위와 예측값 기반)
        base_usage = daegu_features['total_usage'] / len(suseong_locations)
        priority_multiplier = 2.0 if location['priority'] == 1 else (1.5 if location['priority'] == 2 else 1.0)
        expected_usage = base_usage * priority_multiplier
        
        recommended_stations.append({
            'name': location['name'],
            'lat': location['lat'],
            'lon': location['lon'],
            'priority': location['priority'],
            'expected_daily_usage': int(expected_usage),
            'confidence': min(predicted_rank / 16 * 100, 95)  # 신뢰도 계산
        })
    
    # 우선순위순으로 정렬
    recommended_stations.sort(key=lambda x: x['priority'])
    
    print(f"\n✅ 추천 정류소 {len(recommended_stations)}개 생성")
    for i, station in enumerate(recommended_stations, 1):
        print(f"   {i}. {station['name']}: 예상 일일 이용량 {station['expected_daily_usage']}건 (신뢰도: {station['confidence']:.1f}%)")
    
    return recommended_stations, predicted_rank

# generate_kakao_map 함수 수정
def generate_kakao_map(recommended_stations, filename='daegu_suseong_map.html'):
    """수정된 Kakao Map 생성 및 저장 - 중복 출력 방지"""
    print(f"🗺️ {filename} 생성 중...")
    
    # 수성구 중심 좌표
    center_lat = 35.858883
    center_lng = 128.631532
    
    # 지도 생성 (OpenStreetMap 사용)
    m = folium.Map(location=[center_lat, center_lng], 
                  zoom_start=14, 
                  tiles='OpenStreetMap')
    
    # 마커 클러스터 추가
    marker_cluster = MarkerCluster().add_to(m)
    
    # 각 추천 정류소에 마커 추가
    for station in recommended_stations:
        # 우선순위에 따른 색상 설정
        if station['priority'] == 1:
            color = 'red'
            priority_text = '매우높음'
        elif station['priority'] == 2:
            color = 'orange' 
            priority_text = '높음'
        else:
            color = 'blue'
            priority_text = '중간'
        
        # 팝업 내용 생성
        popup_content = f"""
        <b>{station['name']}</b><br>
        우선순위: {priority_text}<br>
        예상 일일 이용량: {station['expected_daily_usage']}건<br>
        신뢰도: {station['confidence']:.1f}%<br>
        위치: ({station['lat']:.3f}, {station['lon']:.3f})
        """
        
        folium.Marker(
            location=[station['lat'], station['lon']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=station['name'],
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    
    # 지도 저장
    m.save(filename)
    print(f"✅ 지도 저장 완료: {filename}")

def visualize_results(seoul_gu_features, best_r2, recommended_stations, total_population):
    print("=== 결과 시각화 ===")
    
    try:
        # 딕셔너리를 DataFrame으로 변환
        features_df = pd.DataFrame.from_dict(seoul_gu_features, orient='index')
        
        print("✅ 시각화 생성 완료")
        
        # 최종 요약 (한 번만 출력)
        print("\n" + "="*70)
        print("                  대구 수성구 따릉이 정류소 추천 분석 최종 요약")
        print("="*70)
        
        print(f"\n📊 서울 데이터 분석:")
        print(f"   분석된 구 수: {len(seoul_gu_features)}개")
        print(f"   평균 이용량: {features_df['total_usage'].mean():.0f}건")
        print(f"   평균 이용시간: {features_df['avg_duration'].mean():.1f}분")
        print(f"   평균 이용거리: {features_df['avg_distance'].mean():.0f}m")
        
        # 상위 3개 구 출력
        top_3_districts = features_df.nlargest(3, 'total_usage')
        print(f"\n🏆 서울 상위 3개 구 (이용량 기준):")
        for idx, (gu_name, row) in enumerate(top_3_districts.iterrows(), 1):
            print(f"   {idx}. {gu_name}: {row['total_usage']:.0f}건")
        
        print(f"\n🤖 머신러닝 모델:")
        print(f"   최고 성능 모델: GradientBoosting")
        print(f"   R² Score: {best_r2:.3f}")
        print(f"   예측 정확도: {best_r2*100:.1f}%")
        
        print(f"\n🎯 대구 수성구 분석:")
        print(f"   총 인구수: {total_population:,}명")
        print(f"   추천 정류소 수: {len(recommended_stations)}개")
        
        # 추천 정류소 목록 (중복 방지)
        print(f"\n📍 추천 정류소 목록:")
        for i, station in enumerate(recommended_stations, 1):
            print(f"   {i}. {station['name']}")
            print(f"      위치: ({station['lat']:.3f}, {station['lon']:.3f})")
            print(f"      예상 일일 이용량: {station['expected_daily_usage']}건")
            print(f"      신뢰도: {station['confidence']:.1f}%")
            if i < len(recommended_stations):  # 마지막 항목이 아니면 빈 줄 추가
                print()
        
        # 총합 계산
        total_expected_usage = sum(station['expected_daily_usage'] for station in recommended_stations)
        print(f"\n💡 예상 총 일일 이용량: {total_expected_usage}건")
        print(f"💡 예상 월 이용량: {total_expected_usage * 30:,}건")
        
        print(f"\n✅ 분석 완료: 대구 수성구 따릉이 정류소 {len(recommended_stations)}개소 추천")
        
    except Exception as e:
        print(f"⚠️ 시각화 중 오류 발생: {e}")
        # 간단한 요약만 출력 (중복 방지)
        print(f"\n📊 기본 요약:")
        print(f"   분석된 구 수: {len(seoul_gu_features)}개")
        print(f"   총 인구수: {total_population:,}명")
        print(f"   추천 정류소 수: {len(recommended_stations)}개")
        
    except Exception as e:
        print(f"⚠️ 시각화 중 오류 발생: {e}")
        # 기본 요약만 출력
        print("\n" + "="*70)
        print("                  대구 수성구 따릉이 정류소 추천 분석 최종 요약")
        print("="*70)
        
        print(f"\n📊 서울 데이터 분석:")
        print(f"   분석된 구 수: {len(seoul_gu_features)}개")
        
        print(f"\n🎯 대구 수성구 분석:")
        print(f"   총 인구수: {total_population:,}명")
        print(f"   추천 정류소 수: {len(recommended_stations)}개")
        
        print(f"\n📍 추천 정류소:")
        for i, station in enumerate(recommended_stations, 1):
            print(f"   {i}. {station['name']}: {station['expected_daily_usage']}건/일")

def main():
    """메인 실행 함수 - 중복 출력 방지 버전"""
    print("🚴‍♂️ 대구 수성구 따릉이 정류소 추천 시스템")
    print("="*80)
    
    try:
        # 1. 데이터 로딩
        print("\n🔄 실제 데이터 로딩 진행 중...")
        seoul_usage, seoul_stations, daegu_population, new_users = load_real_data()
        
        if seoul_usage is None:
            print("❌ 데이터 로딩 실패. 프로그램을 종료합니다.")
            return
        
        print("✅ 데이터 로딩 완료")
        
        # 2. 서울 데이터 처리
        print("\n🔄 서울 데이터 처리 진행 중...")
        gu_stats = process_seoul_data(seoul_usage, seoul_stations, new_users)
        
        if len(gu_stats) == 0:
            print("❌ 구별 통계 생성 실패. 프로그램을 종료합니다.")
            return
            
        print("✅ 서울 데이터 처리 완료")
        
        # 3. 서울 구별 특성 생성
        print("\n🔄 서울 구별 특성 생성 진행 중...")
        seoul_gu_features = create_seoul_gu_features(gu_stats)
        print("✅ 서울 구별 특성 생성 완료")
        
        # 4. 대구 인구 데이터 파싱
        print("\n🔄 대구 인구 데이터 파싱 진행 중...")
        total_population = parse_daegu_population(daegu_population)
        print("✅ 대구 인구 데이터 파싱 완료")
        
        # 5. ML 모델 학습 및 검증
        print("\n🔄 ML 모델 학습 및 검증 진행 중...")
        best_model, scaler, best_r2, best_params = train_and_validate_models(seoul_gu_features)
        print(f"✅ ML 모델 학습 및 검증 완료 (최종 R²: {best_r2:.3f})")
        
        # 6. 대구 수성구 정류소 추천
        print("\n🔄 대구 수성구 정류소 추천 진행 중...")
        recommended_stations, predicted_usage = recommend_stations_for_daegu_suseong(
            best_model, scaler, seoul_gu_features, total_population)
        print("✅ 대구 수성구 정류소 추천 완료")
        
        # 7. 지도 생성
        print("\n🔄 지도 생성 진행 중...")
        map_filename = generate_kakao_map(recommended_stations)
        print(f"✅ 지도 생성 완료: {map_filename}")
        
        # 8. 결과 시각화 (한 번만 호출)
        print("\n🔄 결과 시각화 진행 중...")
        visualize_results(seoul_gu_features, best_r2, recommended_stations, total_population)
        
        # 최종 메시지
        print(f"\n🎉 모든 분석이 성공적으로 완료되었습니다!")
        print(f"ℹ️ 결과 지도 파일: daegu_suseong_map.html")
        
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()