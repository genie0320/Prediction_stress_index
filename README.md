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

### v10 > Local MAE: 0.169968 (타깃 변환 제거 및 강력한 트리 규제 도입)

- [x] **Logit Target Transformation 폐기 및 원본 타깃 공간 복귀**:
  - v8/v9에서 시도된 Logit 변환이 경계값(0과 1) 부근 아웃라이어의 변동폭을 무한대로 폭발시켜 리더보드 일반화에 치명적임을 확인.
  - 타깃 변환을 전면 폐기하고, 원본 타깃 공간($0 \sim 1$)에서 직접 예측을 수행하며 `np.clip` 후처리를 유지하는 안정적인 모델링으로 선회.
- [x] **XGBoost 손실 정규화(`gamma`) 파라미터 도입**:
  - v9의 인위적인 트리 성장 제약(Underfitting 유발)을 극복하기 위해 트리의 구조적 성장 한계는 유지하되, 리프 노드 추가 분할을 제어하는 L2 정규화 복합 손실 가중치 `gamma` (0.0~5.0)를 Optuna 탐색 범위에 정식 추가.
- [x] **SHAP 최하위 피처 제거 기준 확장**:
  - 기존 소심한 3개 피처 제명에서 데이터셋 노이즈 차단을 극대화하기 위해 최하위 20% 피처(6개) 대규모 자동 제거 방식으로 고도화.
- [x] **결측치 시그널 피처 보강**: 
  - `mean_working` 결측 여부 자체를 원-핫 성격의 신호로 학습할 수 있는 `is_working_missing` 플래그를 추가 삽입하여 정보 손실 방어.
- [x] **결과 분석**: 로컬 OOF MAE **0.1699**로, 인위적 규제(v9: 0.1905) 대비 압도적인 개선을 달성함과 동시에 타깃 무한 팽창 리스크를 해결하여 안정적인 최적점에 재안착함.

### v9 > Local MAE: 0.190593 (과소적합 발생 및 교훈)

- [x] **과적합 방어를 위한 하드 제약(Regularization) 실험**: v8의 로컬-리더보드 점수 격차(0.167 vs 0.173)를 트리의 과적합으로 진단. 일반화 성능을 높이기 위해 `max_depth` (3~8) 및 `min_child_weight` (10~50)의 탐색 공간을 강제 축소하여 트리의 물리적 성장을 억제함.
- [x] **기대효과 및 한계 (교훈)**: 미지의 데이터에 대한 안정성(Generalization)을 기대했으나, 규제가 지나치게 강해 오히려 복잡한 생체 피처 간의 상호작용을 모델이 전혀 학습하지 못하는 **극심한 과소적합(Underfitting)**이 발생함. 특히 Logit 변환된 타겟 공간의 극단값을 얕은 트리가 감당하지 못해 0.1905라는 점수 하락을 초래했으며, 이는 물리적 트리 제약 대신 손실함수(`gamma`)를 통한 정석적인 정규화(v10)로 전략을 선회하는 결정적 계기가 됨.
- [x] **XGBoost 내부 버그 원천 우회**: Custom Eval Metric 사용 시 XGBoost 내부 래퍼에서 발생하는 `AttributeError('numpy.ndarray' object has no attribute 'get_label')` 에러를 우회. `eval_metric` 파라미터를 제거하고 Optuna의 `WeightsAndBiasesCallback`에 로깅을 온전히 위임하여 안정적인 튜닝 루프를 완성함.

### v8 > Local MAE: 0.167718 / Public MAE: 0.173078 >> 과적합???

- [x] **앙상블 해체 및 XGBoost 단독 스페셜라이제이션**: v7 결과 가중치가 0으로 수렴했던 Ridge Regression 및 LightGBM을 배제하고, 압도적인 성능을 보인 단일 XGBoost 모델에 연산 및 튜닝 자원을 집중화.
- [x] **Target Logit Transformation 도입**: Target 스트레스 점수(0~1)의 경계 오차(Boundary Error) 문제를 해결하기 위해 학습 데이터 타깃에 $\epsilon(10^{-5})$을 마스킹한 후 `scipy.special.logit`을 적용하여 무한 영역으로 변환 후 학습.
- [x] **정밀 역변환 평가 파이프라인 수립**: Optuna 튜닝 및 Early Stopping의 성능 평가 지표를 Logit 영역이 아닌 원본 영역(Sigmoid/expit 복원)에서 MAE가 계산되도록 설계하여 실제 예측 대상과의 평가지표 정합성 확보.
- [x] **XGBoost 탐색 공간 대폭 스케일업**: Optuna 튜닝 시도를 100회로 확장하고, `max_depth` (3~15), `learning_rate` (0.001~0.1) 등의 탐색 범위를 한계까지 열어 깊은 비선형 패턴 탐색.
- [x] **분석**: 로컬 교차검증(OOF)에서는 **0.1677**로 사상 최저치 MAE를 기록하였으나, 리더보드 점수는 **0.1730**으로 다소 소폭 상승함. 이는 오직 XGBoost 단일 모델에 하이퍼-튜닝을 심하게 집중하면서 발생한 공분산 편향(Overfitting)과, Logit 변환의 아웃라이어 예측 증폭 현상(Sigmoid 복원 시 미세한 오차가 원래 스페이스에서 크게 작용)이 융합된 결과로 판단됨.

### v7 > Target MAE: 0.171128

- [x] **모델 구조조정 (CatBoost 폐기)**: v6 앙상블 결과 가중치 0을 기록하며 자원만 소모하던 CatBoost를 최종 파이프라인에서 완전히 폐기하여 불필요한 연산 낭비 제거.
- [x] **이종 앙상블(Dual-Track Heterogeneous) 아키텍처 도입**: 트리 모델(LightGBM, XGBoost)의 직교 분할 사각지대를 메우고 선형적 관계를 복합 학습하기 위해 선형 규제 모델인 **Ridge Regression**을 메인 모델로 강제 투입.
- [x] **엄격한 Dual-Track 전처리 파이프라인 분리**:
  - **Tree Track**: 범주형 데이터를 원본 범주형(`category` 타입) 그대로 사용하여 트리 분할 정보 보존.
  - **Linear Track (Ridge)**: 선형 모델의 스케일 및 선형성 가정을 만족시키기 위해 K-Fold 내부 루프에서 `RobustScaler`(수치형)와 `OneHotEncoder`(범주형)를 적용하여 데이터 누수(Data Leakage)를 완벽히 통제한 정밀 전처리 구축.
- [x] **튜닝 프로세스 개선**: XGBoost 파이프라인의 내부 전처리 최적화를 통해 단일 XGBoost의 OOF MAE를 기존 0.183에서 **0.171**로 대폭 하락시키는 성능 향상 확보.

### v6 > Target MAE: 0.182317

- [x] **Data-Centric 파이프라인 전환**: 다중공선성 우려로 삭제했던 `height`, `weight`, `systolic_blood_pressure` 등 원본 수치형 데이터를 모두 보존하여 트리의 비선형적 상호작용 학습 극대화.
- [x] **정밀 결측치 대치 (`IterativeImputer`)**: 8개의 수치형 건강 지표를 모두 동원하여 머신러닝 기반으로 `mean_working` 결측치 대치 (Data Leakage 방지를 위해 Train셋으로만 Fit).
- [x] **SHAP 기반 하위 노이즈 제어**: LightGBM Baseline 학습 후 SHAP TreeExplainer 기여도를 산출, 최하위 3개 노이즈 피처 자동 제거를 통해 차원 팽창 방어.
- [x] **Weights & Biases (W&B) 모니터링 연동**: `optuna_integration` 콜백을 통해 3개 GBDT 모델의 하이퍼파라미터 튜닝 시도(각 300회)와 MAE, MSE, RMSE, R2 등 보조 평가지표를 실시간 대시보드에 로깅.

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
