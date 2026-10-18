# One container: Flask serves the API and the built React app on port 5000.

# Stage 1: build the React frontend.
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci
COPY src/frontend ./
RUN npm run build

# Stage 2: Python API with the models and the built frontend.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KAGGLEHUB_CACHE=/data/kagglehub \
    REID_DEVICE=cpu \
    FRONTEND_DIST=/app/frontend

WORKDIR /app/src/backend

# CPU-only PyTorch keeps the image much smaller than the default CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" "torchvision>=0.17"
COPY src/backend/requirements.txt .
RUN pip install -r requirements.txt

# Model weights are Git LFS files: run `git lfs pull` before building.
COPY src/models /app/src/models
RUN python -c "import pathlib, sys; \
small = [str(p) for p in pathlib.Path('/app/src/models').rglob('*.pth') if p.stat().st_size < 1_000_000]; \
sys.exit('Model files are Git LFS pointers, not real weights: ' + ', '.join(small) + '. Run \"git lfs install && git lfs pull\" and rebuild.') if small else None"

COPY src/backend /app/src/backend
COPY --from=frontend /frontend/dist /app/frontend
RUN mkdir -p /data/kagglehub

EXPOSE 5000
CMD ["python", "-c", "from app import create_app; create_app().run(host='0.0.0.0', port=5000)"]
