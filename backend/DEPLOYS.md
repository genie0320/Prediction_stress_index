# 스트레스 지수 측정 서비스 배포 가이드 (Hugging Face Spaces + Netlify)

이 문서는 완성된 프론트엔드와 백엔드를 각각 무료 플랫폼에 배포하여 최종 실서비스를 여는 상세 가이드입니다.

---

## 1단계: 백엔드 API 배포 (Hugging Face Spaces)

Hugging Face Spaces는 Docker SDK를 통해 FastAPI 앱을 24시간 무료로 호스팅할 수 있는 최적의 환경을 제공합니다.

### 1. Space 생성
1. [Hugging Face Spaces](https://huggingface.co/spaces)에 접속하여 로그인합니다.
2. 우측 상단의 **`Create new Space`** 버튼을 클릭합니다.
3. 설정값을 다음과 같이 입력합니다:
   - **Space name**: 자유롭게 설정 (예: `stress-predict-api`)
   - **License**: `mit` (또는 자유 선택)
   - **Select the Space SDK**: **`Docker`**  선택 ⚠️ (중요!)
   - **Choose a Docker template**: **`Blank`** 선택
   - **Space visibility**: `Public` (Netlify와 통신하기 위해 공개 설정 필요)
4. 하단의 **`Create Space`**를 클릭합니다.

### 2. 파일 업로드
새로 만들어진 Space의 `Files and versions` 탭을 통해 `backend/` 폴더 내의 모든 파일들을 업로드합니다.

**업로드할 파일 목록 (총 5개):**
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `light_model_with_bp_svr.pkl`
- `light_model_without_bp_svr.pkl`

*(직접 브라우저에서 `Add file` ➡️ `Upload files`를 누르거나, Git LFS를 이용해 푸시할 수 있습니다. 파일들이 1MB 전후로 아주 가볍기 때문에 브라우저 업로드로 몇 초 만에 전송이 완료됩니다.)*

### 3. 빌드 및 구동 확인
- 업로드가 완료되면 자동으로 Docker 빌드가 시작됩니다. (약 1~2분 소요)
- 빌드가 완료되고 상태가 **`Running`**으로 변하면 정상 구동된 것입니다.
- 본인의 API 서버 공개 엔드포인트 주소는 다음과 같이 구성됩니다:
  `https://[HuggingFace-계정명]-[Space-이름].hf.space`
  - 예시: 계정명이 `chulsoo`이고 Space 이름이 `stress-api`라면:
    `https://chulsoo-stress-api.hf.space`
  - 브라우저에 `https://[본인주소].hf.space/health`를 입력하여 `{"status": "ok", ...}` 결과가 출력되는지 확인해 보세요.

---

## 2단계: 프론트엔드 배포 (Netlify)

Netlify는 HTML/CSS/JS 정적 웹사이트를 클릭 한 번으로 평생 무료 배포할 수 있는 초고속 CDN 서비스입니다.

### 1. API 주소 연동 (로컬 파일 수정)
배포하기 전, [frontend/static/js/app.js](file:///Users/admin/Documents/dev_src/stress_index/frontend/static/js/app.js) 파일의 최상단 API 주소를 방금 생성한 Hugging Face Space의 실제 API 주소로 변경하여 저장합니다.

```javascript
// ⚠️ 수정 전:
const API_BASE_URL = 'http://localhost:7860';

// ⚠️ 수정 후 (본인의 실배포 Space 주소로 입력):
const API_BASE_URL = 'https://[본인계정]-[Space명].hf.space';
```

### 2. Netlify 배포 진행
1. [Netlify](https://www.netlify.com/)에 접속하여 로그인합니다.
2. 로그인 후 대시보드의 **`Add new site`** ➡️ **`Deploy manually`**를 클릭합니다.
3. 드래그 앤 드롭 영역이 나타나면, 프로젝트 내의 **`frontend` 폴더 자체를 통째로 드래그하여 업로드**합니다.
   - ⚠️ **주의**: 폴더 내부 파일들이 아닌 `frontend` 폴더 자체를 드롭해야 합니다.
4. 드롭 즉시 배포가 진행되며 약 5초 만에 완료됩니다.
5. 완료 후 Netlify가 제공하는 임의의 서브도메인 주소(예: `https://handsome-gelato-12345.netlify.app`)를 클릭하여 사이트에 접속합니다.

---

## 3단계: 포트폴리오 연동 및 완료
- 이제 Netlify 주소로 접속하면 모바일, 태블릿, PC 어디서나 작동하는 아름다운 스트레스 측정 웹 서비스가 동작합니다.
- 처음 접속하여 측정을 시작할 때 Hugging Face 서버가 잠들어 있다면(Cold Start), 로딩바 하단에 친절한 설명과 함께 약 15~20초 후 첫 측정이 완료됩니다. 이후 접속 시에는 2초 만에 연산이 즉시 완료됩니다.
- 포트폴리오 메인 사이트에 이 Netlify URL을 링크(Link)로 연결해 주시면 구축이 완벽히 마무리됩니다!
