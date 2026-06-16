# Prediction_stress_index

Predict stress index

## 대회목적 ( 주최 : 데이콘 / 해커톤 )

현대인들의 일상에 깊숙이 자리 잡은 스트레스는 신체적, 정신적 건강에 심각한 영향을 미치고 있으나, 많은 사람들이 자신의 스트레스 수준을 객관적으로 인식하지 못하고 있습니다. 이번 해커톤에서는 신체 정보, 수면 패턴 및 활동량 등 다양한 건강 데이터를 활용하여 개인의 스트레스 점수를 예측하는 AI 알고리즘 개발에 도전하게 됩니다.

[주제]
스트레스 점수 예측 AI 알고리즘 개발

## 대회규칙

1. 평가 : MAE (Public에서 Test 데이터 100%를 활용)

2. 참여 규칙
   개인이나 팀으로 참여 가능

3. 외부 데이터 및 사전 학습 모델
   외부 데이터 사용 불가
   사전 학습 모델(Pre-trained Model) 사용 가능

4. 유의 사항

- 1일 최대 제출 횟수: 3회
- 모델 학습에서 평가 데이터셋 활용(Data Leakage)시 수상 제외
  - label encoding, one-hot encoding 시 test 데이터 셋 활용
  - data scaling 적용 시 test 데이터 셋 활용
  - test 데이터 셋에 pd.get_dummies() 함수 적용
  - test 데이터 셋의 결측치 처리 시 test 데이터 셋의 통계 값 활용
- 위 예시 외에도 test 데이터 셋이 모델 학습에 활용되는 경우에 Data leakage에 해당됨

## 데이터구조

- 특성 수 : 16
- 전체 훈련 데이터셋 : 3000

| Column Name              | Type    | Description            |
| ------------------------ | ------- | ---------------------- |
| ID                       | object  | 샘플별 고유 ID         |
| gender                   | object  | 성별                   |
| age                      | int64   | 연령                   |
| height                   | float64 | 키(cm)                 |
| weight                   | float64 | 몸무게(kg)             |
| cholesterol              | float64 | 콜레스테롤 수치        |
| systolic_blood_pressure  | int64   | 수축기 혈압            |
| diastolic_blood_pressure | int64   | 이완기 혈압            |
| glucose                  | float64 | 혈당 수치(mg/dL)       |
| bone_density             | float64 | 골밀도(g/cm²)          |
| activity                 | object  | 생활시 운동 강도       |
| smoke_status             | object  | 흡연 상태              |
| medical_history          | object  | 만성질환               |
| family_medical_history   | object  | 가족력                 |
| sleep_pattern            | object  | 수면패턴               |
| edu_level                | float64 | 학력                   |
| mean_working             | float64 | 1주일당 평균 근로 시간 |
| stress_score (target)    | float64 | (TARGET) 스트레스 점수 |

<br>
## Change log

### v5 > MAE: 0.183198

- [x] **GBDT 3대장 앙상블 체제 구축**: LightGBM 단일 모델에서 벗어나 XGBoost(`enable_categorical=True`, `tree_method='hist'`), CatBoost(`cat_features` 지정) 등 3대 GBDT 모델 동시 적용 및 개별 하이퍼파라미터 튜닝 환경 구축.
- [x] **최종 모델 시드 앙상블 (Seed Averaging)**: 각 모델마다 서로 다른 3개 시드(`42`, `2026`, `777`)로 5-Fold 최종 학습을 진행하여 모델 분할 편향 제어.
- [x] **SLSQP 기반 블렌딩 가중치 최적화**: 3대 모델의 OOF 예측값을 바탕으로 가중치의 합=1.0, 범위 [0, 1]의 수학적 제약 하에서 MAE가 최소화되는 결합 가중치 최적 산출.
- [x] **Boundary Clipping 후처리**: 최종 예측 점수 범위를 목표 범위인 `[0.0, 1.0]` 내로 제한(`np.clip`)하여 예측 안정성 확보.

### v4 > MAE: 0.188906

- [x] **다차원 결측치 대치 Cascade**: `mean_working` 결측치 대치를 나이대+학력+활동량 3D 그룹 중앙값 우선 대치 및 Fallback 알고리즘으로 정밀화.
- [x] **피처 가지치기 및 다중공선성 통제**: `Ponderal Index(PI)` 채택에 따른 중복 `bmi` 파생 변수 생략 및 중간 이진 플래그 일괄 제거.

### v3 > MAE: 0.184199

- [x] 평가 산식 및 손실 함수 MAE(L1) 일치: `objective='regression_l1'`, `metric='mae'`, `mean_absolute_error`를 전체 파이프라인에 통일하여 절대 오차 최적화.
- [x] 복합 텍스트 범주형 변수의 이진 해체: `medical_history` 및 `family_medical_history` 내 쉼표(`,`) 구분 복합 질환 리스트를 split하여 질환별 고유 이진 원-핫 플래그 생성.
- [x] 고위험 교차 상호작용 피처: 극단과로+수면장애(`overwork_and_poor_sleep`), 저골밀도+고맥압(`vascular_bone_risk`) 플래그 생성.
- [x] 생체 화학 지표 도메인 결합: 혈당과 콜레스테롤 비선형 비율인 `glucose_cholesterol_ratio` 생성.
- [x] 데이터 누수 없는 K-Fold 내 Robust Scaling\*\*: `metabolic_load_index` 생성 시 K-Fold 루프 내부에서 Scaler를 매번 초기화하고 Train 폴드만 `fit`하도록 설계하여 Data Leakage 차단. 판다스 차원 경고 방지를 위한 `.flatten()` 처리 적용.
- [x] 통합 전처리 파이프라인 리팩토링: 파편화되어 마크다운에 갇히거나 순서가 꼬여있던 전처리/피처 엔지니어링 과정을 단일한 순차 코드 블록으로 완전 통합.
- [x] Optuna 자동 하이퍼파라미터 튜닝 도입 : 20회 튜닝 최적화를 통해 예측 성능 대폭 개선 (5-Fold 평균 MAE 0.2225 -> 0.1841)

### v2 > MAE: 0.222509

- [ ] 파라미터 튜닝 : `lr = 0.1`, `num_leaves = 63`, `min_child_samples = 10`, `n_est = 300`
- [ ] 순서형 변수(Ordinal Encoding) 수동 매핑:
  - `edu_level`: Unknown(0) < High school(1) < Bachelor's(2) < Graduate(3)
  - `activity`: Light(1) < Moderate(2) < Intense(3)
- [ ] 이상치 분포 기반 고위험 플래그(이진변수) 추가 :
  - `is_extreme_overwork`: 평균 근무 시간 12시간 이상(극단적과로) 여부, 이상치 보존
  - `is_low_bone_density`: 골밀도 T-Score 의학적 저골밀도 임계치(-1.0)이하여부
  - `is_high_pulse_pressure`: 맥압 80 mmHg 초과 여부(수축기 단독 고혈압을 의심)
- [x] 평균 근무 시간 (`mean_working`) 그룹핑:
  1. 결측 지시자 피처 생성 (`is_working_missing`): 평균 근무 시간이 유무로 `1` / `0`무직·은퇴자 집단 정보 보존
  2. 연령대별 중앙값 대치: 10세 단위 연령대(`age_group`) 그룹의 근무 시간 중앙값으로 세분화하여 결측치를 대치
- [x] 파생변수

### v1 > RMSE: 0.243725

- [x] 하이퍼파라미터 튜닝 : 학습 용량 확대를 위해 LightGBM 모델 파라미터를 최적화했습니다(`lr=0.1`, `num_leaves=63`, `n_est=200`, `random_state=42`).
- [x] 5-Fold Cross Validation (KFold) 도입
- [x] LightGBM이 범주형 패턴을 트리 노드 분할 최적화 : 대상 변수들을 `category` 타입으로 일괄 강제 선언.
- [x] 학력 (`edu_level`): 데이터 수집 과정에서의 정보 부재 > 'Unknown'
- [x] 기존 병력 및 가족력 (`medical_history`, `family_medical_history`): 기왕력이 없는 건강한 사람들을 의미 > 'none'
- [x] 도메인 기반 파생변수 생성 v1
  - 체질량지수 (BMI) : $$\text{BMI} = \frac{\text{weight}}{(\text{height} / 100)^2}$$
  - 맥압 (Pulse Pressure, PP) : 심장이 수축할 때와 이완할 때 혈관이 받는 압력 차이로, 심혈관계 피로도를 반영.
    $$\text{PP} = \text{systolic\_blood\_pressure} - \text{diastolic\_blood\_pressure}$$
  - 평균혈압 (Mean Arterial Pressure, MAP) 생성 : 일정한 주기 동안 전체 혈관 계통에 가해지는 평균 압력으로 신체 스트레스 부하를 대변.
    $$\text{MAP} = \frac{\text{systolic\_blood\_pressure} + 2 \times \text{diastolic\_blood\_pressure}}{3}$$
