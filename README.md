# Prediction_stress_index

Predict stress index

## 대회목적 ( 주최 : 데이콘 / 해커톤 )

## 대회규칙

## 데이터구조

## Change log

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
