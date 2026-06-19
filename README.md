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

### for Web service

### [for Modeling](Docs/Change_log_for_Modeling.md)
