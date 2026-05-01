#!/bin/bash
# Quick Start Script for AWS Deployment

set -e

echo "🚀 Financial AI Platform - AWS Deployment Setup"
echo "================================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi
echo "✅ AWS CLI installed"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✅ Docker installed"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Run:"
    echo "   aws configure"
    exit 1
fi
echo "✅ AWS credentials configured"

# Get AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")

echo ""
echo "📋 Deployment Configuration:"
echo "   AWS Account: $AWS_ACCOUNT_ID"
echo "   AWS Region: $AWS_REGION"
echo ""

# Prompt for confirmation
read -p "Do you want to create AWS resources? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "📦 Step 1: Creating ECR Repositories..."
aws ecr create-repository \
    --repository-name financial-ai-main \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true 2>/dev/null || echo "Repository financial-ai-main already exists"

aws ecr create-repository \
    --repository-name financial-ai-loan \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true 2>/dev/null || echo "Repository financial-ai-loan already exists"

echo "✅ ECR repositories ready"

echo ""
echo "🏗️  Step 2: Creating ECS Cluster..."
aws ecs create-cluster \
    --cluster-name financial-ai-cluster \
    --region $AWS_REGION 2>/dev/null || echo "Cluster financial-ai-cluster already exists"

echo "✅ ECS cluster ready"

echo ""
echo "📁 Step 3: Creating EFS File System..."
EFS_ID=$(aws efs create-file-system \
    --performance-mode generalPurpose \
    --encrypted \
    --tags Key=Name,Value=financial-ai-data \
    --region $AWS_REGION \
    --query 'FileSystemId' \
    --output text 2>/dev/null || echo "")

if [ -n "$EFS_ID" ]; then
    echo "✅ EFS created: $EFS_ID"
    echo "   Please wait 30 seconds for EFS to become available..."
    sleep 30
else
    echo "⚠️  EFS may already exist. Please check manually."
fi

echo ""
echo "🔒 Step 4: Creating Security Group..."
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)

SG_ID=$(aws ec2 create-security-group \
    --group-name financial-ai-ecs-sg \
    --description "Security group for Financial AI ECS tasks" \
    --vpc-id $DEFAULT_VPC \
    --query 'GroupId' \
    --output text 2>/dev/null || echo "")

if [ -n "$SG_ID" ]; then
    # Add inbound rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8501 \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8502 \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    echo "✅ Security group created: $SG_ID"
else
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=financial-ai-ecs-sg" \
        --query "SecurityGroups[0].GroupId" \
        --output text)
    echo "✅ Using existing security group: $SG_ID"
fi

echo ""
echo "📝 Step 5: Updating deployment configuration..."

# Update deploy script
if [ -f deploy-to-aws.sh ]; then
    sed -i.bak "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" deploy-to-aws.sh
    echo "✅ Updated deploy-to-aws.sh"
fi

# Update task definitions
if [ -f aws-ecs-task-main.json ] && [ -n "$EFS_ID" ]; then
    sed -i.bak "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" aws-ecs-task-main.json
    sed -i.bak "s/fs-XXXXXXXXX/$EFS_ID/g" aws-ecs-task-main.json
    echo "✅ Updated aws-ecs-task-main.json"
fi

if [ -f aws-ecs-task-loan.json ] && [ -n "$EFS_ID" ]; then
    sed -i.bak "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" aws-ecs-task-loan.json
    sed -i.bak "s/fs-XXXXXXXXX/$EFS_ID/g" aws-ecs-task-loan.json
    echo "✅ Updated aws-ecs-task-loan.json"
fi

# Create .env file
if [ ! -f .env ]; then
    cp .env.example .env
    sed -i.bak "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" .env
    sed -i.bak "s/us-east-1/$AWS_REGION/g" .env
    echo "✅ Created .env file"
fi

echo ""
echo "✨ AWS Resources Created Successfully!"
echo ""
echo "📋 Summary:"
echo "   ECR Repositories: financial-ai-main, financial-ai-loan"
echo "   ECS Cluster: financial-ai-cluster"
if [ -n "$EFS_ID" ]; then
    echo "   EFS File System: $EFS_ID"
fi
echo "   Security Group: $SG_ID"
echo "   VPC: $DEFAULT_VPC"
echo ""
echo "📤 Next Steps:"
echo ""
echo "1. Upload your data to EFS (if you created one):"
echo "   - Mount EFS to an EC2 instance"
echo "   - Copy data/actual/ to the mounted directory"
echo ""
echo "2. Build and deploy applications:"
echo "   chmod +x deploy-to-aws.sh"
echo "   ./deploy-to-aws.sh"
echo ""
echo "3. Create ECS services (see AWS_DEPLOYMENT_GUIDE.md)"
echo ""
echo "4. Set up Application Load Balancer (optional)"
echo ""
echo "For detailed instructions, see: AWS_DEPLOYMENT_GUIDE.md"
