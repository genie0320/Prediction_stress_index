// Automatically detect if running locally or deployed in production
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
const API_BASE_URL = isLocal 
  ? 'http://localhost:7860' 
  : 'https://featherlike-stress-index-predictor.hf.space';
let currentMode = 'lite'; // 'lite' or 'normal'

/* ==========================================================================
   DOM Elements Selection
   ========================================================================== */
const tabLite = document.getElementById('tab-lite');
const tabNormal = document.getElementById('tab-normal');
const bpFieldset = document.getElementById('bp-fieldset');
const stressForm = document.getElementById('stress-form');
const submitBtn = document.getElementById('submit-btn');

// Inputs
const ageInput = document.getElementById('age');
const heightInput = document.getElementById('height');
const weightInput = document.getElementById('weight');
const systolicInput = document.getElementById('systolic_blood_pressure');
const diastolicInput = document.getElementById('diastolic_blood_pressure');

// Loading Overlay elements
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStatus = document.getElementById('loading-status');
const progressBar = document.getElementById('progress-bar');
const loadingHint = document.getElementById('loading-hint');

// Modal Elements
const resultDialog = document.getElementById('result-dialog');
const resModelType = document.getElementById('res-model-type');
const resGaugeFill = document.getElementById('res-gauge-fill');
const resScorePct = document.getElementById('res-score-pct');
const resRank = document.getElementById('res-rank');
const resLevelBadge = document.getElementById('res-level-badge');
const resDescription = document.getElementById('res-description');
const resAction = document.getElementById('res-action');
const resSupplement = document.getElementById('res-supplement');

/* ==========================================================================
   CORS Warm-up Ping (Cold Start Mitigation)
   ========================================================================== */
window.addEventListener('DOMContentLoaded', () => {
  console.log("Warming up Hugging Face API server...");
  fetch(`${API_BASE_URL}/health`)
    .then(res => res.json())
    .then(data => console.log("HF Server Active status:", data))
    .catch(err => console.warn("Warm-up ping failed (Server might be sleeping):", err));
});

/* ==========================================================================
   Tab Swapping & Mode Toggle
   ========================================================================== */
tabLite.addEventListener('click', () => setMode('lite'));
tabNormal.addEventListener('click', () => setMode('normal'));

function setMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;
  
  if (mode === 'lite') {
    tabLite.classList.add('active');
    tabNormal.classList.remove('active');
    bpFieldset.classList.add('collapsed');
    bpFieldset.disabled = true;
    
    // Clear validation states for BP inputs
    clearError(systolicInput);
    clearError(diastolicInput);
  } else {
    tabNormal.classList.add('active');
    tabLite.classList.remove('active');
    bpFieldset.classList.remove('collapsed');
    bpFieldset.disabled = false;
  }
}

/* ==========================================================================
   Form Field Validations on Blur / Input
   ========================================================================== */
const validationRules = {
  age: { min: 1, max: 120 },
  height: { min: 50, max: 250 },
  weight: { min: 10, max: 300 },
  systolic: { min: 50, max: 250 },
  diastolic: { min: 30, max: 200 }
};

function validateField(input, rules) {
  const value = parseFloat(input.value);
  const parent = input.closest('.input-group');
  
  if (isNaN(value) || value < rules.min || value > rules.max) {
    parent.classList.add('invalid');
    return false;
  } else {
    parent.classList.remove('invalid');
    return true;
  }
}

function clearError(input) {
  const parent = input.closest('.input-group');
  if (parent) parent.classList.remove('invalid');
}

// Add event listeners for instant validation
ageInput.addEventListener('blur', () => validateField(ageInput, validationRules.age));
heightInput.addEventListener('blur', () => validateField(heightInput, validationRules.height));
weightInput.addEventListener('blur', () => validateField(weightInput, validationRules.weight));
systolicInput.addEventListener('blur', () => {
  if (!bpFieldset.disabled) validateField(systolicInput, validationRules.systolic);
});
diastolicInput.addEventListener('blur', () => {
  if (!bpFieldset.disabled) validateField(diastolicInput, validationRules.diastolic);
});

// Clear invalid state while typing
[ageInput, heightInput, weightInput, systolicInput, diastolicInput].forEach(input => {
  input.addEventListener('input', () => clearError(input));
});

function validateForm() {
  let isValid = true;
  isValid &= validateField(ageInput, validationRules.age);
  isValid &= validateField(heightInput, validationRules.height);
  isValid &= validateField(weightInput, validationRules.weight);
  
  if (currentMode === 'normal') {
    isValid &= validateField(systolicInput, validationRules.systolic);
    isValid &= validateField(diastolicInput, validationRules.diastolic);
  }
  
  return isValid;
}

/* ==========================================================================
   Form Submit & API Call with Progress Bar Simulation
   ========================================================================== */
stressForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  if (!validateForm()) {
    // Focus first invalid element
    const firstInvalid = document.querySelector('.input-group.invalid input');
    if (firstInvalid) firstInvalid.focus();
    return;
  }
  
  // Disable submit button and show loading overlay
  submitBtn.disabled = true;
  loadingOverlay.classList.remove('hidden');
  loadingHint.classList.add('hidden');
  progressBar.style.width = '0%';
  
  // Collect inputs
  const formData = new FormData(stressForm);
  const dataPayload = {
    age: parseInt(formData.get('age')),
    height: parseFloat(formData.get('height')),
    weight: parseFloat(formData.get('weight')),
    gender: formData.get('gender'),
    activity: formData.get('activity'),
    sleep_pattern: formData.get('sleep_pattern'),
    smoke_status: formData.get('smoke_status')
  };
  
  if (currentMode === 'normal') {
    dataPayload.systolic_blood_pressure = parseFloat(formData.get('systolic_blood_pressure'));
    dataPayload.diastolic_blood_pressure = parseFloat(formData.get('diastolic_blood_pressure'));
  }
  
  // Simulate progress bar and state messages
  let progress = 0;
  const statusTexts = [
    { threshold: 0, text: "서버 데이터 전송 중..." },
    { threshold: 25, text: "신체 정보 및 생활 습관 전처리 중..." },
    { threshold: 55, text: "AI 스트레스 모델 연산 진행 중..." },
    { threshold: 80, text: "맞춤형 피드백 가이드 구성 중..." }
  ];
  
  let animationFinished = false;
  let apiResponse = null;
  let apiError = null;
  
  // Start parallel API request
  const apiPromise = fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dataPayload)
  })
  .then(res => {
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return res.json();
  })
  .then(data => {
    apiResponse = data;
  })
  .catch(err => {
    apiError = err;
    console.error("API Error occurred:", err);
  });
  
  // Run progress loading animation (at least 2 seconds)
  const startTime = Date.now();
  const progressInterval = setInterval(() => {
    const elapsed = Date.now() - startTime;
    
    // Smooth progress mapping
    if (elapsed < 2000) {
      progress = (elapsed / 2000) * 90; // Go up to 90% in 2 seconds
    } else if (!apiResponse && !apiError) {
      // API taking longer (Cold Start)
      progress = 90 + ((elapsed - 2000) / 10000) * 8; // Crawl up to 98% over next 10 seconds
      progress = Math.min(progress, 98);
      
      // Reveal cold start hint after 2.5 seconds
      if (elapsed > 2500) {
        loadingHint.classList.remove('hidden');
      }
    } else {
      // API returned, fast forward to 100%
      progress = 100;
      clearInterval(progressInterval);
      animationFinished = true;
      
      setTimeout(() => {
        loadingOverlay.classList.add('hidden');
        submitBtn.disabled = false;
        
        if (apiError) {
          alert("모델 서버와의 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n(서버 Cold Start 상태일 수 있으니 10초 뒤 다시 측정해보세요!)");
        } else if (apiResponse && apiResponse.success) {
          openResultModal(apiResponse);
        } else {
          alert("예측 오류: " + (apiResponse ? apiResponse.message : "알 수 없는 오류"));
        }
      }, 300);
    }
    
    // Update progress bar width
    progressBar.style.width = `${progress}%`;
    
    // Update status text
    const textObj = [...statusTexts].reverse().find(t => progress >= t.threshold);
    if (textObj) loadingStatus.innerText = textObj.text;
    
  }, 50);
});

/* ==========================================================================
   Modal Dialog Binding & Open
   ========================================================================== */
function openResultModal(data) {
  // 1. Model Type Badge
  resModelType.innerText = data.model_type;
  
  // 2. Score Percentage Text
  resScorePct.innerText = `${data.percent_value}%`;
  
  // 3. Percentile Rank Text
  resRank.innerText = `상위 ${data.top_percentile}%`;
  
  // 4. Level Badge Text, CSS mapping, and Gauge Path color
  resLevelBadge.innerText = data.category;
  
  // Reset all classes
  resLevelBadge.className = 'level-badge';
  resLevelBadge.classList.add(data.level_class);
  
  // Map theme colors for gauge circle stroke
  let gaugeColor = 'var(--primary)';
  switch(data.level_class) {
    case 'very-low': gaugeColor = 'var(--color-very-low)'; break;
    case 'low': gaugeColor = 'var(--color-low)'; break;
    case 'normal': gaugeColor = 'var(--color-normal)'; break;
    case 'high': gaugeColor = 'var(--color-high)'; break;
    case 'very-high': gaugeColor = 'var(--color-very-high)'; break;
  }
  resGaugeFill.style.stroke = gaugeColor;
  
  // 5. Circle dash offset animation
  // Circumference is 264. Offset = 264 - (264 * score)
  const scoreRatio = data.score;
  const offset = 264 - (264 * scoreRatio);
  
  // Force reset and delay to trigger SVG transition
  resGaugeFill.style.strokeDashoffset = '264';
  setTimeout(() => {
    resGaugeFill.style.strokeDashoffset = offset.toString();
  }, 100);
  
  // 6. Descriptions & Suggestions
  resDescription.innerText = data.description;
  resAction.innerText = data.action;
  resSupplement.innerText = data.supplement;
  
  // 7. Show native modal dialog
  resultDialog.showModal();
}

// Light dismiss: Close modal if backdrop is clicked
resultDialog.addEventListener('click', (e) => {
  const dialogDimensions = resultDialog.getBoundingClientRect();
  if (
    e.clientX < dialogDimensions.left ||
    e.clientX > dialogDimensions.right ||
    e.clientY < dialogDimensions.top ||
    e.clientY > dialogDimensions.bottom
  ) {
    resultDialog.close();
  }
});
