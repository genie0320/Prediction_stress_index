// Automatically detect if running locally or deployed in production
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
const API_BASE_URL = isLocal 
  ? 'http://localhost:7860' 
  : 'https://featherlike-stress-index-predictor.hf.space';

/* ==========================================================================
   DOM Elements Selection
   ========================================================================== */
const stressForm = document.getElementById('stress-form');
const submitBtn = document.getElementById('submit-btn');

// Inputs
const ageInput = document.getElementById('age');
const heightInput = document.getElementById('height');
const weightInput = document.getElementById('weight');
const useBpCheckbox = document.getElementById('use-bp-checkbox');
const bpInputArea = document.getElementById('bp-input-area');
const systolicInput = document.getElementById('systolic_blood_pressure');
const diastolicInput = document.getElementById('diastolic_blood_pressure');

// Loading Overlay elements
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStatus = document.getElementById('loading-status');
const progressBar = document.getElementById('progress-bar');
const loadingHint = document.getElementById('loading-hint');

// Result Step Elements
const resModelType = document.getElementById('res-model-type');
const resGaugeFill = document.getElementById('res-gauge-fill');
const resScorePct = document.getElementById('res-score-pct');
const resRank = document.getElementById('res-rank');
const resLevelBadge = document.getElementById('res-level-badge');
const resDescription = document.getElementById('res-description');
const resAction = document.getElementById('res-action');
const resSupplement = document.getElementById('res-supplement');
const resetBtn = document.getElementById('reset-btn');

// Wizard elements
const wizardWrapper = document.getElementById('wizard-wrapper');

// Theme Switcher Buttons
const themeLightBtn = document.getElementById('theme-light');
const themeDarkBtn = document.getElementById('theme-dark');

/* ==========================================================================
   Theme Switcher Logic (Apple Light/Dark Style)
   ========================================================================== */
function applyTheme(theme) {
  if (theme === 'dark') {
    document.documentElement.classList.add('dark');
    document.documentElement.classList.remove('light');
    themeDarkBtn.classList.add('active');
    themeLightBtn.classList.remove('active');
  } else {
    document.documentElement.classList.add('light');
    document.documentElement.classList.remove('dark');
    themeLightBtn.classList.add('active');
    themeDarkBtn.classList.remove('active');
  }
}

themeLightBtn.addEventListener('click', () => {
  applyTheme('light');
  localStorage.setItem('theme', 'light');
});

themeDarkBtn.addEventListener('click', () => {
  applyTheme('dark');
  localStorage.setItem('theme', 'dark');
});

// Initialize theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
  applyTheme(savedTheme);
} else {
  // Sync with system preference by default
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(systemPrefersDark ? 'dark' : 'light');
}

// Listen to system preference changes dynamically
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
  if (!localStorage.getItem('theme')) {
    applyTheme(e.matches ? 'dark' : 'light');
  }
});

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
   Wizard Navigation & Step Control
   ========================================================================== */
let currentStep = 0;
let lastResult = null;

function goToStep(step) {
  currentStep = step;
  wizardWrapper.style.transform = `translateX(-${currentStep * 100}%)`;
  
  // Set accessibility focus/attributes if needed
  const activeStepCard = document.getElementById(`step-${step}`);
  if (activeStepCard) {
    activeStepCard.focus();
  }
}

// Setup data-next and data-prev click handlers
document.querySelectorAll('[data-next]').forEach(btn => {
  btn.addEventListener('click', () => {
    const nextStep = parseInt(btn.getAttribute('data-next'));
    // If going from Step 1 (Physical Info) to Step 2 (Lifestyle), validate first
    if (currentStep === 1 && nextStep === 2) {
      if (!validatePhysicalStep()) {
        const firstInvalid = document.querySelector('#step-1 .input-group.invalid input');
        if (firstInvalid) firstInvalid.focus();
        return;
      }
    }
    goToStep(nextStep);
  });
});

document.querySelectorAll('[data-prev]').forEach(btn => {
  btn.addEventListener('click', () => {
    const prevStep = parseInt(btn.getAttribute('data-prev'));
    goToStep(prevStep);
  });
});

/* ==========================================================================
   BP Checkbox Switch Control
   ========================================================================== */
useBpCheckbox.addEventListener('change', () => {
  if (useBpCheckbox.checked) {
    bpInputArea.classList.remove('collapsed');
    systolicInput.disabled = false;
    diastolicInput.disabled = false;
  } else {
    bpInputArea.classList.add('collapsed');
    systolicInput.disabled = true;
    diastolicInput.disabled = true;
    clearError(systolicInput);
    clearError(diastolicInput);
  }
});

/* ==========================================================================
   Form Field Validations
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

// Blur events
ageInput.addEventListener('blur', () => validateField(ageInput, validationRules.age));
heightInput.addEventListener('blur', () => validateField(heightInput, validationRules.height));
weightInput.addEventListener('blur', () => validateField(weightInput, validationRules.weight));
systolicInput.addEventListener('blur', () => {
  if (useBpCheckbox.checked) validateField(systolicInput, validationRules.systolic);
});
diastolicInput.addEventListener('blur', () => {
  if (useBpCheckbox.checked) validateField(diastolicInput, validationRules.diastolic);
});

// Input events to clear visual error state
[ageInput, heightInput, weightInput, systolicInput, diastolicInput].forEach(input => {
  input.addEventListener('input', () => clearError(input));
});

function validatePhysicalStep() {
  let isValid = true;
  isValid &= validateField(ageInput, validationRules.age);
  isValid &= validateField(heightInput, validationRules.height);
  isValid &= validateField(weightInput, validationRules.weight);
  
  if (useBpCheckbox.checked) {
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
  
  // Make sure Physical step info is valid (just in case they hacked steps)
  if (!validatePhysicalStep()) {
    goToStep(1);
    const firstInvalid = document.querySelector('#step-1 .input-group.invalid input');
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
  
  // Check if BP checkbox is ticked, if so include BP params to route to Model A (With BP)
  if (useBpCheckbox.checked) {
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
          lastResult = apiResponse;
          showResultStep(apiResponse);
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
   Result Step Binding & Presentation
   ========================================================================== */
function showResultStep(data) {
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
  }, 150);
  
  // 6. Descriptions & Suggestions
  resDescription.innerText = data.description;
  resAction.innerText = data.action;
  resSupplement.innerText = data.supplement;
  
  // 7. Transition to Step 3 (Results)
  goToStep(3);
}

/* ==========================================================================
   Reset / Measure Again Action
   ========================================================================== */
function resetFormAndGoIntro() {
  stressForm.reset();
  
  // Explicitly ensure BP inputs are collapsed and disabled
  bpInputArea.classList.add('collapsed');
  systolicInput.disabled = true;
  diastolicInput.disabled = true;
  
  // Clear any validation visual states
  [ageInput, heightInput, weightInput, systolicInput, diastolicInput].forEach(clearError);
  
  // Clear share state
  lastResult = null;
  
  // Go back to intro
  goToStep(0);
}

// Navigation & Sharing Event Listeners (Step 3 & 4)
const goToShareBtn = document.getElementById('go-to-share-btn');
const backToResultsBtn = document.getElementById('back-to-results-btn');
const resetBtnShare = document.getElementById('reset-btn-share');

resetBtn.addEventListener('click', () => {
  resetFormAndGoIntro();
});

goToShareBtn.addEventListener('click', () => {
  goToStep(4);
});

backToResultsBtn.addEventListener('click', () => {
  goToStep(3);
});

resetBtnShare.addEventListener('click', () => {
  resetFormAndGoIntro();
});

/* ==========================================================================
   Share Page Setup & Channels (Kakao, X, copy link, system share)
   ========================================================================== */
const webShareBtn = document.getElementById('web-share-btn');
const shareKakao = document.getElementById('share-kakao');
const shareTwitter = document.getElementById('share-twitter');
const shareCopyLink = document.getElementById('share-copy-link');
const shareToast = document.getElementById('share-toast');

let toastTimeout = null;

function showToast(message) {
  shareToast.innerText = message;
  shareToast.classList.remove('hidden');
  
  if (toastTimeout) clearTimeout(toastTimeout);
  
  toastTimeout = setTimeout(() => {
    shareToast.classList.add('hidden');
  }, 2500);
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.error("Clipboard copy failed: ", err);
    // Fallback using older method if navigator.clipboard fails
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
    } catch (e) {
      alert("공유 문구가 복사되지 못했습니다. 직접 복사해주세요:\n" + text);
    }
    document.body.removeChild(textArea);
  }
}

function getShareData() {
  const shareUrl = window.location.origin + window.location.pathname;
  const shareText = lastResult 
    ? `[Stress Index Lab] 제 스트레스 분석 결과는 '상위 ${lastResult.top_percentile}% (${lastResult.category})' 입니다. AI 맞춤형 분석 가이드를 직접 확인해보세요!`
    : '[Stress Index Lab] 개인 맞춤형 AI 스트레스 지수 분석기';
  return { title: 'Stress Index Lab', text: shareText, url: shareUrl };
}

// System Web Share API Support check
if (navigator.share) {
  webShareBtn.addEventListener('click', async () => {
    const shareData = getShareData();
    try {
      await navigator.share(shareData);
    } catch (err) {
      console.warn("Web Share cancelled or failed:", err);
    }
  });
} else {
  // Hide system share button if unsupported
  webShareBtn.style.display = 'none';
}

// Kakao fallback (Copy text + link)
shareKakao.addEventListener('click', async () => {
  const data = getShareData();
  const fullText = `${data.text}\n측정 링크: ${data.url}`;
  await copyToClipboard(fullText);
  showToast("카톡 공유용 결과가 클립보드에 복사되었습니다! 💬");
});

// Twitter sharing
shareTwitter.addEventListener('click', () => {
  const data = getShareData();
  const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(data.text)}&url=${encodeURIComponent(data.url)}`;
  window.open(twitterUrl, '_blank', 'noopener,noreferrer');
});

// Link copy only
shareCopyLink.addEventListener('click', async () => {
  const shareUrl = window.location.origin + window.location.pathname;
  await copyToClipboard(shareUrl);
  showToast("공유 링크가 클립보드에 복사되었습니다! 📋");
});
