# Build the React frontend.
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY src/frontend ./
RUN npm ci && npm run build

# Flask serves the API and the built frontend on port 5000.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 REID_DEVICE=cpu FRONTEND_DIST=/app/frontend KAGGLEHUB_CACHE=/data/kagglehub
WORKDIR /app/src/backend

# CPU-only PyTorch keeps the image small.
COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/models /app/src/models
COPY src/backend .
COPY --from=frontend /frontend/dist /app/frontend

EXPOSE 5000
CMD ["python", "-c", "from app import create_app; create_app().run(host='0.0.0.0', port=5000)"]
