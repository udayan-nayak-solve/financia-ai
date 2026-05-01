#!/bin/bash
# AWS ECS Deployment Script for Financial AI Platform
# This script builds and deploys both applications to AWS ECS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - UPDATE THESE VALUES
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-YOUR_ACCOUNT_ID}"
ECR_REPO_MAIN="${ECR_REPO_MAIN:-financial-ai-main}"
ECR_REPO_LOAN="${ECR_REPO_LOAN:-financial-ai-loan}"
ECS_CLUSTER="${ECS_CLUSTER:-financial-ai-cluster}"
ECS_SERVICE_MAIN="${ECS_SERVICE_MAIN:-financial-ai-main-service}"
ECS_SERVICE_LOAN="${ECS_SERVICE_LOAN:-financial-ai-loan-service}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Financial AI Platform - AWS Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check AWS credentials
echo -e "${YELLOW}Checking AWS credentials...${NC}"
aws sts get-caller-identity > /dev/null 2>&1 || {
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    exit 1
}
echo -e "${GREEN}✓ AWS credentials configured${NC}"

# Login to ECR
echo -e "${YELLOW}Logging in to Amazon ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo -e "${GREEN}✓ Logged in to ECR${NC}"

# Build and push Main Platform
echo -e "${YELLOW}Building Main Platform Docker image...${NC}"
docker build -t $ECR_REPO_MAIN:latest .
docker tag $ECR_REPO_MAIN:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_MAIN:latest
docker tag $ECR_REPO_MAIN:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_MAIN:$(date +%Y%m%d-%H%M%S)

echo -e "${YELLOW}Pushing Main Platform image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_MAIN:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_MAIN:$(date +%Y%m%d-%H%M%S)
echo -e "${GREEN}✓ Main Platform image pushed${NC}"

# Build and push Loan Prediction System
echo -e "${YELLOW}Building Loan Prediction Docker image...${NC}"
cd loan_prediction_system
docker build -t $ECR_REPO_LOAN:latest .
docker tag $ECR_REPO_LOAN:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_LOAN:latest
docker tag $ECR_REPO_LOAN:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_LOAN:$(date +%Y%m%d-%H%M%S)

echo -e "${YELLOW}Pushing Loan Prediction image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_LOAN:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_LOAN:$(date +%Y%m%d-%H%M%S)
cd ..
echo -e "${GREEN}✓ Loan Prediction image pushed${NC}"

# Update ECS services
echo -e "${YELLOW}Updating ECS services...${NC}"

# Force new deployment for Main Platform
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE_MAIN \
    --force-new-deployment \
    --region $AWS_REGION \
    > /dev/null 2>&1 || echo -e "${YELLOW}Note: Main service may need to be created first${NC}"

# Force new deployment for Loan Prediction
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE_LOAN \
    --force-new-deployment \
    --region $AWS_REGION \
    > /dev/null 2>&1 || echo -e "${YELLOW}Note: Loan service may need to be created first${NC}"

echo -e "${GREEN}✓ ECS services updated${NC}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Check deployment status:"
echo -e "  aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE_MAIN $ECS_SERVICE_LOAN --region $AWS_REGION"
