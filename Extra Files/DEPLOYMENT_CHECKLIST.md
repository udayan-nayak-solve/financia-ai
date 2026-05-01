# AWS Deployment Checklist

## ✅ Files Created for AWS Deployment

### Docker Configuration
- [x] `Dockerfile` (main platform) - Already exists
- [x] `loan_prediction_system/Dockerfile` - **NEW** - Created for loan prediction
- [x] `.dockerignore` (main) - Already exists  
- [x] `loan_prediction_system/.dockerignore` - **NEW**
- [x] `docker-compose.yml` - **UPDATED** - Added loan prediction service
- [x] `requirements-prod.txt` - **NEW** - Production dependencies with pinned versions
- [x] `loan_prediction_system/requirements-prod.txt` - **NEW**

### AWS ECS Configuration
- [x] `aws-ecs-task-main.json` - **NEW** - Task definition for main platform
- [x] `aws-ecs-task-loan.json` - **NEW** - Task definition for loan prediction
- [x] `deploy-to-aws.sh` - **NEW** - Automated deployment script
- [x] `setup-aws-resources.sh` - **NEW** - Initial AWS setup script

### Documentation
- [x] `AWS_DEPLOYMENT_GUIDE.md` - **NEW** - Complete deployment guide
- [x] `DOCKER_GUIDE.md` - **NEW** - Docker usage guide
- [x] `.env.example` - **NEW** - Environment configuration template

## 📝 Pre-Deployment Checklist

### 1. AWS Account Setup
- [ ] AWS account created and accessible
- [ ] AWS CLI installed: `aws --version`
- [ ] AWS credentials configured: `aws configure`
- [ ] Appropriate IAM permissions for ECS, ECR, EFS, EC2

### 2. Local Prerequisites
- [ ] Docker installed: `docker --version`
- [ ] Docker daemon running
- [ ] Git repository up to date
- [ ] All dependencies tested locally

### 3. Configuration Files
- [ ] Copy `.env.example` to `.env`
- [ ] Update AWS_ACCOUNT_ID in `.env`
- [ ] Update AWS_REGION in `.env`
- [ ] Review and update `config/config.yaml` if needed

### 4. Data Preparation
- [ ] Verify `data/models/` contains trained models
- [ ] Verify `data/outputs/` contains necessary outputs
- [ ] Plan data upload strategy for `data/actual/` (312MB - too large for Docker image)
  - Option A: Upload to EFS
  - Option B: Upload to S3 and download on container start
  - Option C: Mount from external storage

### 5. Security Review
- [ ] Review security group rules (ports 8501, 8502)
- [ ] Consider VPC configuration (public vs private subnets)
- [ ] Plan for SSL/TLS certificates (if using custom domain)
- [ ] Review IAM roles and policies
- [ ] Consider enabling AWS WAF for additional protection

## 🚀 Deployment Steps

### Quick Deployment (Automated)

```bash
# 1. Set up AWS resources (one-time)
./setup-aws-resources.sh

# 2. Deploy applications
./deploy-to-aws.sh
```

### Manual Deployment (Step-by-Step)

See `AWS_DEPLOYMENT_GUIDE.md` for detailed instructions.

#### Phase 1: Create AWS Infrastructure
```bash
# 1. Create ECR repositories
aws ecr create-repository --repository-name financial-ai-main
aws ecr create-repository --repository-name financial-ai-loan

# 2. Create ECS cluster
aws ecs create-cluster --cluster-name financial-ai-cluster

# 3. Create EFS file system (for data persistence)
aws efs create-file-system --performance-mode generalPurpose

# 4. Create security groups
# See AWS_DEPLOYMENT_GUIDE.md for details

# 5. Create IAM roles
# See AWS_DEPLOYMENT_GUIDE.md for details
```

#### Phase 2: Build and Push Images
```bash
# 1. Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# 2. Build main platform
docker build -t financial-ai-main .
docker tag financial-ai-main:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-main:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-main:latest

# 3. Build loan prediction
cd loan_prediction_system
docker build -t financial-ai-loan .
docker tag financial-ai-loan:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-loan:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/financial-ai-loan:latest
```

#### Phase 3: Create ECS Services
```bash
# 1. Register task definitions
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-main.json
aws ecs register-task-definition --cli-input-json file://aws-ecs-task-loan.json

# 2. Create services
# See AWS_DEPLOYMENT_GUIDE.md for details

# 3. Set up Application Load Balancer (optional)
# See AWS_DEPLOYMENT_GUIDE.md for details
```

## 🔍 What Changed in Existing Files

### Modified: `docker-compose.yml`
**Change:** Added `loan-prediction` service configuration
**Impact:** Can now run both applications together locally
**Action Required:** None - backward compatible

### Modified: `requirements.txt` 
**Change:** Cleaned up duplicates, organized by category
**Impact:** Cleaner dependency management
**Action Required:** None - all dependencies preserved

### Modified: `loan_prediction_system/requirements.txt`
**Change:** Added missing dependencies, organized structure
**Impact:** Complete dependency list for deployment
**Action Required:** None - verified against code

## 📊 No Changes Needed to Application Code

✅ **No changes required** to any Python source files!

All applications will work exactly as they do now. The deployment files are pure infrastructure:
- Dockerfiles define how to package the apps
- ECS task definitions define how to run on AWS
- Scripts automate the deployment process
- Guides document the process

Your Python code in `src/` and `loan_prediction_system/src/` remains unchanged.

## 🎯 Key Points About the Deployment

### Architecture
```
Internet
    ↓
Application Load Balancer (optional)
    ↓
┌─────────────────┬─────────────────┐
│  Main Platform  │ Loan Prediction │
│   ECS/Fargate   │   ECS/Fargate   │
│   Port 8501     │   Port 8502     │
└────────┬────────┴────────┬────────┘
         └────────┬─────────┘
                  ↓
         EFS File System
         (Shared Data)
```

### Resource Requirements

**Main Platform:**
- CPU: 2 vCPU (2048 units)
- Memory: 4 GB (4096 MB)
- Storage: EFS for data/actual/ (~300MB)
- Port: 8501

**Loan Prediction:**
- CPU: 1 vCPU (1024 units)
- Memory: 2 GB (2048 MB)
- Storage: Models included in image (~10MB)
- Port: 8502

### Estimated Monthly Costs

**AWS Fargate:**
- Main Platform: ~$35/month (24/7 operation)
- Loan Prediction: ~$18/month (24/7 operation)

**Other Services:**
- Application Load Balancer: ~$20/month (optional)
- EFS Storage: ~$5/month (17GB at $0.30/GB)
- CloudWatch Logs: ~$1/month
- Data Transfer: Variable (typically $1-5/month)

**Total: ~$80-100/month** for both applications running 24/7

**Cost Optimization:**
- Use Fargate Spot: Save 70% (but may be interrupted)
- Auto-scaling: Only run when needed
- Schedule: Turn off during off-hours

### Data Handling Strategy

Since `data/actual/` is **312MB** and excluded from git:

**Option 1: EFS (Recommended)**
```bash
# One-time upload to EFS
# Mount EFS to EC2 instance
# Copy data/actual/ to mounted directory
# ECS tasks will mount same EFS
```

**Option 2: S3 + Download on Start**
```bash
# Upload to S3
aws s3 cp data/actual/ s3://your-bucket/data/actual/ --recursive

# Modify Dockerfile CMD to download on start:
CMD ["sh", "-c", "aws s3 sync s3://your-bucket/data/actual/ /app/data/actual/ && streamlit run ..."]
```

**Option 3: Bake into Docker Image (Not Recommended)**
- Image size would be ~350MB+
- Slower deployments
- Not following best practices

## ✅ Testing Before Deployment

### Test Locally with Docker

```bash
# Test main platform
docker build -t financial-ai-main .
docker run -p 8501:8501 -v $(pwd)/data:/app/data financial-ai-main
# Access: http://localhost:8501

# Test loan prediction
cd loan_prediction_system
docker build -t financial-ai-loan .
docker run -p 8502:8502 -v $(pwd)/models:/app/models financial-ai-loan
# Access: http://localhost:8502

# Test with docker-compose
docker-compose up
# Access both applications
```

### Test Production Requirements

```bash
# Use production requirements
docker build -f Dockerfile --build-arg REQUIREMENTS=requirements-prod.txt -t financial-ai-main:prod .
```

## 🆘 Support Resources

- **AWS Documentation**: https://docs.aws.amazon.com/ecs/
- **Docker Documentation**: https://docs.docker.com/
- **Streamlit Deployment**: https://docs.streamlit.io/deploy
- **This Project's Guides**:
  - `AWS_DEPLOYMENT_GUIDE.md` - Detailed AWS deployment
  - `DOCKER_GUIDE.md` - Docker usage and troubleshooting
  - `README.md` - Application overview

## 📞 Need Help?

Common issues and solutions:

1. **"Cannot connect to Docker daemon"**
   - Start Docker Desktop
   - Run: `docker ps` to verify

2. **"AWS credentials not found"**
   - Run: `aws configure`
   - Enter your access key and secret

3. **"Permission denied" on scripts**
   - Run: `chmod +x deploy-to-aws.sh setup-aws-resources.sh`

4. **"Port already in use"**
   - Stop other containers: `docker-compose down`
   - Check processes: `lsof -i :8501`

5. **"Out of memory" during build**
   - Increase Docker memory limit in Docker Desktop settings
   - Use multi-stage builds

## 🎉 Next Steps After Deployment

1. **Set up custom domain** (optional)
   - Register domain with Route 53
   - Create SSL certificate with ACM
   - Configure ALB with HTTPS

2. **Implement authentication** (optional)
   - Use AWS Cognito
   - Add SSO integration
   - Implement API keys

3. **Set up monitoring**
   - CloudWatch dashboards
   - Custom metrics
   - Alerts for errors

4. **Implement CI/CD**
   - AWS CodePipeline
   - GitHub Actions
   - Automated testing

5. **Data backup strategy**
   - EFS snapshots
   - S3 versioning
   - Regular backups

## 📝 Summary

**What You Have:**
- ✅ Production-ready Docker containers
- ✅ AWS ECS task definitions
- ✅ Automated deployment scripts
- ✅ Complete documentation
- ✅ Environment configuration
- ✅ Separate deployments for each app

**What You Need to Do:**
1. Set up AWS account and credentials
2. Run `setup-aws-resources.sh` (one-time)
3. Upload data to EFS or S3
4. Run `deploy-to-aws.sh`
5. Access your applications via ALB or public IPs

**Time Estimate:**
- AWS setup: 30-60 minutes (first time)
- Deployment: 10-15 minutes
- Testing and configuration: 30 minutes

**Skill Level Required:**
- Basic: Use automated scripts
- Intermediate: Customize configuration
- Advanced: Modify infrastructure as needed
