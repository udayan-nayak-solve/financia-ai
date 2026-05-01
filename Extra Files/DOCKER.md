# 🐳 Docker Deployment Guide - Financial AI Platform

This guide explains how to deploy and run the Financial AI Platform using Docker.

## 📋 Prerequisites

- Docker Desktop or Docker Engine (20.10+)
- Docker Compose (2.0+)
- At least 4GB of available RAM
- 2GB of free disk space

**Installation Check:**
```bash
docker --version
docker-compose --version
```

## 🚀 Quick Start

### Option 1: Using the Helper Script (Recommended)

Make the script executable:
```bash
chmod +x docker-run.sh
```

Build and run:
```bash
./docker-run.sh build    # Build the image
./docker-run.sh up       # Start the main dashboard
```

Access the dashboard at: **http://localhost:8501**

### Option 2: Manual Docker Commands

Build the image:
```bash
docker build -t financial-ai:latest .
```

Run the container:
```bash
docker run -d \
  --name financial-ai-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  financial-ai:latest
```

### Option 3: Docker Compose (Production)

Start all services:
```bash
docker-compose up -d
```

With YoY Analysis service:
```bash
docker-compose --profile yoy up -d
```

## 📊 Available Services

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| Main Dashboard | 8501 | http://localhost:8501 | Primary analytics dashboard |
| YoY Analysis | 8502 | http://localhost:8502 | Year-over-Year analysis (optional) |

## 🛠️ Common Commands

### Using docker-run.sh Script

```bash
./docker-run.sh build      # Build Docker image
./docker-run.sh up         # Start main dashboard
./docker-run.sh yoy        # Start with YoY analysis
./docker-run.sh stop       # Stop all services
./docker-run.sh logs       # View logs
./docker-run.sh shell      # Open shell in container
./docker-run.sh clean      # Remove all containers and images
./docker-run.sh rebuild    # Full rebuild and restart
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d                      # Main dashboard only
docker-compose --profile yoy up -d        # Include YoY analysis

# Stop services
docker-compose down                       # Stop and remove containers
docker-compose down -v                    # Also remove volumes

# View logs
docker-compose logs -f                    # All services
docker-compose logs -f dashboard          # Specific service

# Restart services
docker-compose restart                    # Restart all
docker-compose restart dashboard          # Restart specific service

# Scale services (if needed)
docker-compose up -d --scale dashboard=2
```

### Using Docker CLI

```bash
# List running containers
docker ps

# View logs
docker logs -f financial-ai-dashboard

# Execute command in container
docker exec -it financial-ai-dashboard bash

# Stop container
docker stop financial-ai-dashboard

# Remove container
docker rm financial-ai-dashboard

# View resource usage
docker stats financial-ai-dashboard
```

## 📂 Volume Mounts

The application uses persistent volumes for data storage:

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./data` | `/app/data` | HMDA data files, models, outputs |
| `./logs` | `/app/logs` | Application and training logs |
| `./config` | `/app/config` | Configuration files |

**Important:** Ensure these directories exist and contain required data before starting:

```bash
# Required data files
data/actual/2022_state_KS.csv
data/actual/2023_state_KS.csv
data/actual/2024_state_KS.csv
data/actual/enhanced_census_data.csv

# Optional but recommended
config/config.yaml
```

## 🔧 Configuration

### Environment Variables

You can override settings using environment variables in `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO
```

### Custom Configuration

Edit `config/config.yaml` for application-specific settings:

```yaml
model_config:
  test_size: 0.2
  random_state: 42
  
data_paths:
  hmda_data: "data/actual"
  models: "data/models"
  outputs: "data/outputs"
```

## 🏥 Health Checks

The containers include automatic health checks:

```bash
# Check container health
docker ps

# If unhealthy, view logs
docker logs financial-ai-dashboard

# Inspect health check
docker inspect --format='{{json .State.Health}}' financial-ai-dashboard | python -m json.tool
```

**Health Check Details:**
- Endpoint: `http://localhost:8501/_stcore/health`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs financial-ai-dashboard

# Check if port is already in use
lsof -i :8501

# Remove existing container and try again
docker rm -f financial-ai-dashboard
./docker-run.sh up
```

### Data Not Loading

```bash
# Verify volume mounts
docker inspect financial-ai-dashboard | grep Mounts -A 20

# Check file permissions
ls -la data/actual/

# Access container shell to debug
docker exec -it financial-ai-dashboard bash
ls -la /app/data/actual/
```

### Out of Memory

```bash
# Increase Docker memory limit (Docker Desktop)
# Settings -> Resources -> Memory -> 4GB or more

# Check memory usage
docker stats financial-ai-dashboard

# Restart with memory limit
docker run -m 4g --name financial-ai-dashboard ...
```

### Module Not Found Errors

```bash
# Rebuild without cache
docker build --no-cache -t financial-ai:latest .

# Check installed packages in container
docker exec financial-ai-dashboard pip list

# Verify requirements file
cat requirements-docker.txt
```

### Dashboard Shows "Please wait..."

This is normal during initial load. Wait 30-60 seconds for:
- Data loading (290K+ records)
- Model initialization
- Feature engineering

Monitor logs:
```bash
docker logs -f financial-ai-dashboard
```

## 📈 Performance Optimization

### Build Cache

Use Docker build cache for faster rebuilds:
```bash
# Rebuild only changed layers
docker build -t financial-ai:latest .

# Force full rebuild
docker build --no-cache -t financial-ai:latest .
```

### Multi-Stage Build (Future Enhancement)

Current Dockerfile can be optimized with multi-stage builds:
```dockerfile
# Builder stage
FROM python:3.11 as builder
RUN pip install --user -r requirements-docker.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

### Resource Limits

Set resource limits in `docker-compose.yml`:
```yaml
services:
  dashboard:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## 🔒 Security Best Practices

### 1. Don't Run as Root

The Dockerfile creates a non-root user:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Scan for Vulnerabilities

```bash
# Scan image for vulnerabilities
docker scan financial-ai:latest

# Use Snyk or Trivy for deeper scans
trivy image financial-ai:latest
```

### 3. Network Security

```bash
# Run on internal network only
docker network create financial-ai-network
docker run --network financial-ai-network ...
```

### 4. Secrets Management

Don't hardcode secrets. Use Docker secrets:
```bash
echo "my_api_key" | docker secret create api_key -
```

## 📦 Production Deployment

### Using Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  dashboard:
    image: financial-ai:latest
    restart: always
    ports:
      - "80:8501"
    volumes:
      - /data/financial-ai:/app/data:ro  # Read-only
      - financial-ai-logs:/app/logs
    environment:
      - ENV=production
      - LOG_LEVEL=WARNING
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
```

### Deploy to Cloud

**AWS ECS:**
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker tag financial-ai:latest <ecr-url>/financial-ai:latest
docker push <ecr-url>/financial-ai:latest
```

**Google Cloud Run:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/financial-ai
gcloud run deploy financial-ai --image gcr.io/PROJECT_ID/financial-ai --port 8501
```

**Azure Container Instances:**
```bash
# Create container
az container create \
  --resource-group financial-ai-rg \
  --name financial-ai \
  --image financial-ai:latest \
  --ports 8501
```

## 🔄 Updates and Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
./docker-run.sh rebuild

# Or manually
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup Data

```bash
# Backup volumes
docker run --rm \
  -v financial-ai_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data

# Backup database (if applicable)
docker exec financial-ai-dashboard pg_dump -U postgres > backup.sql
```

### Clean Up Old Images

```bash
# Remove dangling images
docker image prune

# Remove all unused images
docker image prune -a

# Remove stopped containers
docker container prune
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Docker Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Application README](./README.md)
- [YoY Analysis Guide](./ENHANCED_YOY_ANALYSIS_README.md)

## 🆘 Support

If you encounter issues:

1. **Check logs:** `docker logs -f financial-ai-dashboard`
2. **Verify data files:** Ensure all CSV files are in `data/actual/`
3. **Check health:** `docker ps` and look for "healthy" status
4. **Restart services:** `./docker-run.sh stop && ./docker-run.sh up`
5. **Full rebuild:** `./docker-run.sh clean && ./docker-run.sh build && ./docker-run.sh up`

For persistent issues, include these in your bug report:
```bash
# System info
docker version
docker-compose version
uname -a

# Container info
docker logs financial-ai-dashboard > error.log
docker inspect financial-ai-dashboard > inspect.json
```

---

**Happy Deploying! 🚀**
