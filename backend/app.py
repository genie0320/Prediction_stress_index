import os
import math
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Stress Index Prediction API", version="1.0.0")

# CORS middleware to allow requests from Netlify and other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the SVR ensemble models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_A_PATH = os.path.join(BASE_DIR, "light_model_with_bp_svr.pkl")
MODEL_B_PATH = os.path.join(BASE_DIR, "light_model_without_bp_svr.pkl")

try:
    model_a = joblib.load(MODEL_A_PATH)
    print("Successfully loaded Model A (With BP)")
except Exception as e:
    model_a = None
    print(f"Failed to load Model A: {e}")

try:
    model_b = joblib.load(MODEL_B_PATH)
    print("Successfully loaded Model B (Without BP)")
except Exception as e:
    model_b = None
    print(f"Failed to load Model B: {e}")

class StressInput(BaseModel):
    age: int = Field(..., example=35)
    height: float = Field(..., example=175.0)
    weight: float = Field(..., example=72.0)
    gender: str = Field(..., example="M")  # 'F' or 'M'
    activity: str = Field(..., example="moderate")  # 'light', 'moderate', 'intense'
    sleep_pattern: str = Field(..., example="normal")  # 'sleep difficulty', 'normal', 'oversleeping'
    smoke_status: str = Field(..., example="non-smoker")  # 'non-smoker', 'ex-smoker', 'current-smoker'
    systolic_blood_pressure: Optional[float] = Field(None, example=120.0)
    diastolic_blood_pressure: Optional[float] = Field(None, example=80.0)

def preprocess_input(data: StressInput, use_bp: bool):
    """
    Preprocess raw input dictionary to match features used during training.
    """
    user_dict = {
        'age': data.age,
        'height': data.height,
        'weight': data.weight,
        'gender': data.gender,
        'activity': data.activity,
        'sleep_pattern': data.sleep_pattern,
        'smoke_status': data.smoke_status
    }
    
    if use_bp:
        user_dict['systolic_blood_pressure'] = data.systolic_blood_pressure
        user_dict['diastolic_blood_pressure'] = data.diastolic_blood_pressure
        
    df = pd.DataFrame([user_dict])
    
    # Calculate BMI
    df['bmi'] = df['weight'] / (df['height'] / 100.0) ** 2
    
    # Calculate BP indicators if using BP
    if use_bp:
        df['pp'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
        df['map_val'] = (df['systolic_blood_pressure'] + 2 * df['diastolic_blood_pressure']) / 3
    else:
        df['pp'] = 0.0
        df['map_val'] = 0.0
        
    # Map categoricals to encodings
    df['activity_enc'] = df['activity'].map({'light': 1, 'moderate': 2, 'intense': 3}).fillna(0).astype(int)
    df['sleep_enc'] = df['sleep_pattern'].map({'sleep difficulty': 1, 'normal': 2, 'oversleeping': 3}).fillna(0).astype(int)
    df['gender_enc'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0).astype(int)
    df['smoke_enc'] = df['smoke_status'].map({'non-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}).fillna(0).astype(int)
    
    return df

def predict_ensemble(preprocessed_df, model_package):
    """
    Run inference on the 5-fold SVR ensemble and return final score in [0, 1].
    """
    features = model_package['features']
    svr_pipelines = model_package['svr_pipelines']
    
    X_features = preprocessed_df[features]
    
    svr_preds = []
    for pipe in svr_pipelines:
        sc = pipe['scaler']
        qt = pipe['qt']
        svr = pipe['svr']
        
        # Scaling & prediction
        X_scaled = sc.transform(X_features)
        pred_t = svr.predict(X_scaled)
        
        # QuantileTransformer inverse transform
        pred = qt.inverse_transform(pred_t.reshape(-1, 1)).flatten()
        svr_preds.append(np.clip(pred, 0, 1))
        
    # Average predictions
    final_score = np.mean(svr_preds, axis=0)
    final_score = np.clip(final_score, 0, 1)
    
    return float(final_score[0])

def normal_cdf(x, mean=0.5, std=0.15):
    """
    Compute normal distribution CDF using math.erf.
    """
    if std <= 0:
        return 0.5
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

def get_recommendation(score: float):
    """
    Get stress level classification, description, and recommendations.
    """
    percent_val = round(score * 100, 1)
    cdf_val = normal_cdf(score, mean=0.5, std=0.15)
    top_percentile = round((1.0 - cdf_val) * 100, 1)
    
    # Clip top percentile
    top_percentile = max(0.1, min(99.9, top_percentile))
    
    if score <= 0.20:
        category = "매우 낮음"
        level_class = "very-low"
        description = "스트레스가 매우 낮은 아주 평온한 상태입니다. 마음의 여유를 즐기며 긍정적인 에너지를 유지하세요."
        action = "현재 상태를 유지하며 가벼운 산책이나 취미 생활을 즐겨보세요."
        supplement = "충분한 수분 섭취와 비타민 C"
    elif score <= 0.35:
        category = "낮음"
        level_class = "low"
        description = "스트레스가 낮은 편으로 안정적인 상태를 유지하고 있습니다. 가벼운 휴식으로 몸과 마음을 리프레시해 보세요."
        action = "스트레칭을 하거나 가벼운 명상으로 일상의 긴장을 풀어주세요."
        supplement = "그린티(L-테아닌)나 마그네슘"
    elif score <= 0.65:
        category = "보통"
        level_class = "normal"
        description = "지극히 정상적이고 일상적인 스트레스 수준입니다. 업무나 공부 중간에 적절한 휴식을 취하는 것이 좋습니다."
        action = "규칙적인 수면과 함께 가벼운 운동으로 일상적인 피로를 해소해 보세요."
        supplement = "비타민 B 복합체나 홍경천 추출물"
    elif score <= 0.80:
        category = "높음"
        level_class = "high"
        description = "스트레스 지수가 다소 높게 측정되었습니다. 피로가 누적되지 않도록 적극적인 휴식과 기분 전환이 필요한 때입니다."
        action = "심호흡을 하고 스마트폰을 멀리한 채 따뜻한 목욕이나 조용한 음악 감상을 권장합니다."
        supplement = "아슈와간다 또는 오메가-3"
    else:
        category = "매우 높음"
        level_class = "very-high"
        description = "당신은 현재 스트레스가 매우 높게 측정되고 있습니다. 마음챙김, 보조식품 등으로 자신을 적극적으로 돌보는 시간을 가져보세요."
        action = "즉시 하던 일을 멈추고 깊은 휴식을 취하거나, 전문적인 상담 및 명상 시간을 가져보세요."
        supplement = "L-테아닌 고함량 제품이나 락티움"
        
    return {
        "score": round(score, 4),
        "percent_value": percent_val,
        "top_percentile": top_percentile,
        "category": category,
        "level_class": level_class,
        "description": description,
        "action": action,
        "supplement": supplement
    }

@app.get("/health")
def health_check():
    """
    Endpoint for server warm-up and monitoring.
    """
    return {
        "status": "ok",
        "models_loaded": {
            "model_a": model_a is not None,
            "model_b": model_b is not None
        }
    }

@app.post("/predict")
def predict(data: StressInput):
    # Determine if we use BP model (Model A)
    use_bp = data.systolic_blood_pressure is not None and data.diastolic_blood_pressure is not None
    
    # Check if correct model is loaded
    selected_model = model_a if use_bp else model_b
    if selected_model is None:
        return {
            "success": False,
            "message": "Selected prediction model is not loaded on the server."
        }
        
    # Preprocess
    df_preprocessed = preprocess_input(data, use_bp)
    
    # Predict
    score = predict_ensemble(df_preprocessed, selected_model)
    
    # Generate recommendations
    response_data = get_recommendation(score)
    response_data["success"] = True
    response_data["model_type"] = "Normal (With BP)" if use_bp else "Lite (Without BP)"
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces port defaults to 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
