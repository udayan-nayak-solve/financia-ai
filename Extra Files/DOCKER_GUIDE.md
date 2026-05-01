# Docker Deployment Guide

## Quick Start - Local Development

### Option 1: Using Docker Compose (Recommended)

Run all services together:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access the applications:
- Main Platform: http://localhost:8501
- Loan Prediction: http://localhost:8502
- YoY Analysis (optional): `docker-compose --profile yoy up`

### Option 2: Individual Containers

**Main Platform:**
```bash
docker build -t financial-ai-main .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  financial-ai-main
```

**Loan Prediction:**
```bash
cd loan_prediction_system
docker build -t financial-ai-loan .
docker run -p 8502:8502 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  financial-ai-loan
```

## Building Images

### Main Platform
```bash
# Standard build
docker build -t financial-ai-main:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t financial-ai-main:latest .

# Build with cache disabled (slower but ensures fresh build)
docker build --no-cache -t financial-ai-main:latest .
```

### Loan Prediction System
```bash
cd loan_prediction_system
docker build -t financial-ai-loan:latest .
```

## Environment Variables

Set environment variables in docker-compose.yml or pass with `-e`:

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e DATA_DIR=/app/data/actual \
  -e LOG_LEVEL=INFO \
  financial-ai-main
```

## Volume Mounts

### Main Platform
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  financial-ai-main
```

### Loan Prediction
```bash
docker run -p 8502:8502 \
  -v $(pwd)/loan_prediction_system/models:/app/models \
  -v $(pwd)/loan_prediction_system/logs:/app/logs \
  financial-ai-loan
```

## Health Checks

Check container health:

```bash
# Check status
docker ps

# View health check logs
docker inspect --format='{{json .State.Health}}' financial-ai-dashboard | jq

# Manual health check
curl http://localhost:8501/_stcore/health
```

## Troubleshooting

### View Logs
```bash
# Docker Compose
docker-compose logs -f financial-ai-dashboard
docker-compose logs -f loan-prediction

# Individual Container
docker logs -f container-name
```

### Access Container Shell
```bash
# Docker Compose
docker-compose exec financial-ai-dashboard /bin/bash

# Individual Container
docker exec -it container-name /bin/bash
```

### Check Resource Usage
```bash
docker stats
```

### Clean Up
```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi financial-ai-main financial-ai-loan

# Clean up all unused resources
docker system prune -a
```

## Production Deployment

### Multi-stage Build (Smaller Images)

Create `Dockerfile.prod`:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements-prod.txt .
RUN pip install --user --no-cache-dir -r requirements-prod.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8501
CMD ["streamlit", "run", "src/executive_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

Build:
```bash
docker build -f Dockerfile.prod -t financial-ai-main:prod .
```

### Security Best Practices

1. **Non-root user:**
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

2. **Read-only filesystem:**
```bash
docker run --read-only -p 8501:8501 financial-ai-main
```

3. **Resource limits:**
```bash
docker run -p 8501:8501 \
  --memory="2g" \
  --cpus="1.5" \
  financial-ai-main
```

## Docker Registry

### Push to Docker Hub
```bash
# Tag image
docker tag financial-ai-main:latest your-username/financial-ai-main:latest

# Login
docker login

# Push
docker push your-username/financial-ai-main:latest
```

### Push to AWS ECR
```bash
# Login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag financial-ai-main:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-main:latest

# Push
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-main:latest
```

## Performance Optimization

### Use BuildKit
```bash
DOCKER_BUILDKIT=1 docker build -t financial-ai-main .
```

### Layer Caching
Order Dockerfile commands from least to most frequently changing:
1. System dependencies
2. Python requirements
3. Application code

### Reduce Image Size
- Use alpine base images (if compatible)
- Multi-stage builds
- Remove build dependencies
- Exclude unnecessary files with .dockerignore

## Monitoring

### Container Stats
```bash
docker stats financial-ai-dashboard loan-prediction
```

### Export Logs
```bash
docker logs financial-ai-dashboard > app.log 2>&1
```

### Health Check Endpoint
```bash
while true; do
  curl -f http://localhost:8501/_stcore/health || echo "Health check failed"
  sleep 30
done
```
