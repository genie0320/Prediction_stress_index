# Change_log_for_Modeling

### v18 > Local MAE: 0.13460 (ExtraTrees 분류기 기반 방향 보정 도입) / LB : 0.14104

- [x] **극단값 예측 강화를 위한 방향 보정(Directional Correction) 도입**:
  - SVR 회귀 모델이 중간값으로 예측치를 수축시켜 극단값 영역($y < 0.15$ 및 $y > 0.85$)에서 오차가 커지는 문제를 해결하기 위해, 극단값을 탐지하는 3-class ExtraTrees 분류기(OOF AUC 0.789)를 도입.
  - 분류기의 예측 확률을 기반으로 양방향 신호(`direction = P(high) - P(low)`)를 생성하여 SVR 예측값에 추가 보정: $\text{corrected} = \text{SVR\_pred} + \alpha \times \text{direction}$.
  - Grid Search를 통해 최적의 보정 가중치 $\alpha = 0.02$ 도출 (다중시드 적용 시 최종 보정).
- [x] **초경량화 패키징 구현**:
  - 토이 서비스 배포를 위해 1-Seed \* 5-Fold 구성의 경량 ET Classifier 및 SVR 앙상블 모델과 전처리 스케일러를 `joblib` 압축 패키징(`light_model_with_bp.pkl` 등)하여 용량을 극적으로 축소.
- [x] **결과 분석**: 방향 보정을 적용하여 10-Fold 3-Seed SVR 기준선(0.13470) 대비 OOF MAE **0.13460**으로 추가 성능 개선 달성.

### v17 > Local MAE: 0.13470 (역방향 피처 제거 및 SVR 앙상블 안정화)

- [x] **Backward Feature Elimination을 통한 최적 피처셋 선정**:
  - 기존 19개 피처에 대해 후진 제거법을 적용한 결과, `mean_working_filled`(결측치를 0으로 대치한 근무시간) 변수를 제거할 때 교차검증 오차가 유의미하게 감소하는 것을 확인. 이를 제거하여 **18개 코어 피처셋**으로 최적화.
- [x] **Calibration 및 후처리 기법 검증**:
  - 이산화된 타깃 분포에 맞춰 Isotonic Regression, Variance-Adjusted Calibration, Quantile Transformer(uniform)를 통한 히스토그램 매칭 등을 테스트했으나, 과적합 및 일반화 실패로 최종 MAE가 악화됨을 확인하여 원본 SVR 예측의 단순 반올림(`np.round(..., 2)`) 후처리를 유지.
- [x] **결과 분석**: 18개 피처셋 기반 10-Fold 3-Seed SVR 앙상블을 적용하여 OOF MAE **0.13470**으로 성능 최적화.

### v16 > Local MAE: 0.14044 (SVR 하이퍼파라미터 극대화 및 MLP 앙상블) / LB : 0.14482

- [x] **SVR 하이퍼파라미터 광범위 튜닝**:
  - Optuna를 통해 1000회 이상의 대규모 탐색(Extreme Tuning)을 진행하여 최적 파라미터 조합 `C=28.6022`, `gamma=1.0566`, `n_quantiles=1365` 도출.
- [x] **SVR + MLP 이종 앙상블 실험**:
  - SVR 단독 성능 한계를 극복하기 위해 동일 전처리 피처 기반의 Multi-Layer Perceptron(MLP) 회귀 모델을 구현하고 SVR과 가중 평균 앙상블(SVR 0.8 : MLP 0.2) 실험 진행.
- [x] **Permutation Feature Importance 분석**:
  - SVR 파이프라인 상에서 19개 피처의 순열 중요도를 평가하여 불필요한 노이즈 변수 탐색의 기초 마련.
- [x] **결과 분석**: 대규모 튜닝과 다중 시드(3-Seed) 앙상블을 결합하여 10-Fold 교차검증 MAE **0.14044**(앙상블 단독 MAE 기준 0.13470)를 달성하며 성능 대폭 리드.

### v15 > Local MAE: 0.13951 (SVR 모델 최초 도입 및 범주형 직교 전처리)

- [x] **SVR (Support Vector Regressor) 파이프라인 구축**:
  - 트리 모델 위주의 구성에서 벗어나 연속형/비선형 패턴 학습에 강한 RBF 커널 SVR 모델을 메인으로 최초 도입.
  - target 분포 변환을 위해 `TransformedTargetRegressor`와 `QuantileTransformer(normal)`를 통합하여 타깃 공간을 정규분포화 후 학습.
- [x] **SVR 최적화 전처리 파이프라인 재설계**:
  - SVR이 직교적인 변수 관계를 학습할 수 있도록 모든 범주형/순서형 변수를 One-Hot Encoding으로 변환. 수치형 피처는 `RobustScaler`로 스케일링.
- [x] **결과 분석**: Optuna 최적 하이퍼파라미터(`C=3.56`, `gamma=0.83`, `n_quantiles=983`)를 적용하고 5-Seed 10-Fold 앙상블 및 소수점 둘째 자리 반올림을 결합하여 OOF MAE **0.13951**을 기록, 기존 트리 모델 대비 압도적 성능 도약.

### v14 > Local MAE: 0.18349 (극한 깊이 GBDT 및 순위 보정 실험)

- [x] **극한의 Tree Depth GBDT 실험**:
  - ExtraTrees의 깊은 분할을 모방하기 위해 LightGBM의 `num_leaves=2047`, `max_depth=15` 등 극한 파라미터를 적용하고 앙상블(`ExtraTrees 0.7 + LightGBM 0.3`) 구성.
- [x] **Rank Calibration (순위 보정) 도입**:
  - 예측값의 순서(Rank)를 유지한 채 Train셋의 실제 타깃 분포와 1:1 매칭하거나 Test 예측을 균등 구간(`np.linspace(0, 1)`)으로 매핑하는 강제 이산화 기법 테스트.
- [x] **결과 분석**: 순위 보정을 통해 로컬 검증 점수가 대폭 하락하는 양상을 보였으나 리더보드 일반화에는 한계가 있어, 앙상블 단독 기준 OOF MAE **0.18349** 기록.

### v13 > Local MAE: 0.22 (KNN 기반 이산화 탐색)

- [x] **KNN (K-Nearest Neighbors) 회귀 모델 도입**:
  - 타깃 변수 `stress_score`가 소수점 둘째 자리 단위의 고도로 이산화된 분포를 가지는 특징을 포착하여, KNN ($k=1, 3, 5$) 모델을 적용하여 이산값 예측 성능 검증.
- [x] **ExtraTrees 예측값과의 상관성 비교**:
  - KNN ($k=1$) 예측값과 v12 ExtraTrees 예측값의 상관계수(Pearson $r \approx 0.77$) 분석을 통해 트리 모델과 이웃 모델 간의 상호작용 평가.

### v12 > Local MAE: 0.18526 (ExtraTrees 및 RandomForest 벤치마크)

- [x] **다양한 트리 계열 모델 벤치마크**:
  - LightGBM, XGBoost 외에 Bagging 계열인 ExtraTrees와 RandomForest 모델을 추가하고 기본 8개 피처셋 상에서 성능 비교.
- [x] **ExtraTrees 하이퍼파라미터 그리드 서치**:
  - 분할 기준(`squared_error` vs `absolute_error`), `max_features` 비율, 부트스트랩 여부 등을 튜닝하여 ExtraTrees 최적 파라미터 탐색.
- [x] **결과 분석**: 8개 피처셋 기준 단독 ExtraTrees OOF MAE **0.18526** 달성.

### v11 > Local MAE: 0.17020 (Logit L1 Loss 튜닝 및 Quantile Rounding 도입)

- [x] **Logit 타깃 공간의 L1 Loss 학습 안정화**:
  - XGBoost 튜닝 시 무한대로 발산할 수 있는 Logit 변환 공간에서 `objective="reg:absoluteerror"` (LightGBM은 `regression_l1`)를 강제 적용하여 에러 폭발 문제를 방지하고 튜닝 최적화 루프 정상화.
- [x] **[BREAKTHROUGH] Quantile Rounding (반올림 후처리) 도입**:
  - 스트레스 점수가 0.01 단위로 조밀하게 분절되어 있는 점을 활용해, 최종 예측값을 소수점 둘째 자리까지 반올림(`np.round(..., 2)`)하는 후처리를 최초 도입하여 MAE 소폭 개선.
- [x] **SHAP Whitelist 및 결측치 시그널 보완**:
  - SHAP 기여도 0.0 기준 피처 제거 시, 도메인 파생 변수들을 Whitelist로 보호하여 학습 정보 손실을 최소화. `IterativeImputer` 적용.
- [x] **결과 분석**: XGBoost + LightGBM 앙상블로 OOF MAE **0.17020** 기록.

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
