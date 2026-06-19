---
title: Stress Index Predictor
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Stress Index Predictor API

FastAPI-based backend API for predicting stress index using lightweight 5-fold SVR models.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 7860
```
