# AWS Deployment Guide for Financial AI Platform

This guide covers deploying both applications (Main Platform and Loan Prediction System) to AWS using ECS with Fargate.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Load Balancer            │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ Target Group (8501)  │  │ Target Group (8502)  │   │
│  └──────────────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   ECS Service (Main)    │  │  ECS Service (Loan)     │
│   - Fargate Task        │  │   - Fargate Task        │
│   - Port 8501           │  │   - Port 8502           │
│   - 2 vCPU / 4GB RAM   │  │   - 1 vCPU / 2GB RAM   │
└─────────────────────────┘  └─────────────────────────┘
              │                           │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   EFS File System   │
              │   - data/actual/    │
              │   - data/models/    │
              │   - data/outputs/   │
              └─────────────────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed locally
4. **Domain name** (optional, for custom URL)

## Step 1: Create AWS Resources

### 1.1 Create ECR Repositories

```bash
# Create repository for main platform
aws ecr create-repository \
    --repository-name financial-ai-main \
    --region us-east-1 \
    --image-scanning-configuration scanOnPush=true

# Create repository for loan prediction
aws ecr create-repository \
    --repository-name financial-ai-loan \
    --region us-east-1 \
    --image-scanning-configuration scanOnPush=true
```

### 1.2 Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name financial-ai-cluster \
    --region us-east-1 \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy \
        capacityProvider=FARGATE,weight=1 \
        capacityProvider=FARGATE_SPOT,weight=1
```

### 1.3 Create EFS File System (for data persistence)

```bash
# Create EFS file system
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --encrypted \
    --tags Key=Name,Value=financial-ai-data \
    --region us-east-1

# Note the FileSystemId from output (fs-XXXXXXXXX)
# Update this in aws-ecs-task-*.json files
```

### 1.4 Create VPC and Security Groups (if needed)

```bash
# Get default VPC
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)

# Create security group for ECS tasks
aws ec2 create-security-group \
    --group-name financial-ai-ecs-sg \
    --description "Security group for Financial AI ECS tasks" \
    --vpc-id $DEFAULT_VPC

# Allow inbound traffic on ports 8501 and 8502
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=financial-ai-ecs-sg" --query "SecurityGroups[0].GroupId" --output text)

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8502 \
    --cidr 0.0.0.0/0
```

### 1.5 Create IAM Roles

Create `ecsTaskExecutionRole` if it doesn't exist:

```bash
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file://ecs-task-execution-role-trust-policy.json

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

Trust policy file (`ecs-task-execution-role-trust-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## Step 2: Upload Data to EFS

```bash
# Mount EFS to EC2 instance or local machine
# Then copy data
aws s3 cp data/actual/ s3://your-bucket/data/actual/ --recursive
# Or use EFS mount helper to copy directly to EFS
```

## Step 3: Build and Push Docker Images

### Option A: Using deployment script (Recommended)

```bash
# Update configuration in deploy-to-aws.sh
export AWS_ACCOUNT_ID="123456789012"
export AWS_REGION="us-east-1"

# Make script executable
chmod +x deploy-to-aws.sh

# Run deployment
./deploy-to-aws.sh
```

### Option B: Manual deployment

```bash
# Set variables
AWS_ACCOUNT_ID="123456789012"
AWS_REGION="us-east-1"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push main platform
docker build -t financial-ai-main:latest .
docker tag financial-ai-main:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/financial-ai-main:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/financial-ai-main:latest

# Build and push loan prediction
cd loan_prediction_system
docker build -t financial-ai-loan:latest .
docker tag financial-ai-loan:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/financial-ai-loan:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/financial-ai-loan:latest
cd ..
```

## Step 4: Register Task Definitions

Update the task definition files with your AWS Account ID and EFS file system ID, then register:

```bash
# Update task definitions with your values
sed -i '' "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" aws-ecs-task-main.json
sed -i '' "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/g" aws-ecs-task-loan.json
sed -i '' "s/fs-XXXXXXXXX/$EFS_FILE_SYSTEM_ID/g" aws-ecs-task-main.json
sed -i '' "s/fs-XXXXXXXXX/$EFS_FILE_SYSTEM_ID/g" aws-ecs-task-loan.json

# Register task definitions
aws ecs register-task-definition \
    --cli-input-json file://aws-ecs-task-main.json \
    --region us-east-1

aws ecs register-task-definition \
    --cli-input-json file://aws-ecs-task-loan.json \
    --region us-east-1
```

## Step 5: Create Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
    --name financial-ai-alb \
    --subnets subnet-XXXXX subnet-YYYYY \
    --security-groups $SG_ID \
    --scheme internet-facing \
    --type application

# Create target groups
aws elbv2 create-target-group \
    --name financial-ai-main-tg \
    --protocol HTTP \
    --port 8501 \
    --vpc-id $DEFAULT_VPC \
    --target-type ip \
    --health-check-path /_stcore/health

aws elbv2 create-target-group \
    --name financial-ai-loan-tg \
    --protocol HTTP \
    --port 8502 \
    --vpc-id $DEFAULT_VPC \
    --target-type ip \
    --health-check-path /_stcore/health
```

## Step 6: Create ECS Services

```bash
# Get subnet IDs and security group ID
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

# Create Main Platform service
aws ecs create-service \
    --cluster financial-ai-cluster \
    --service-name financial-ai-main-service \
    --task-definition financial-ai-main-platform \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers targetGroupArn=$MAIN_TG_ARN,containerName=financial-ai-dashboard,containerPort=8501

# Create Loan Prediction service
aws ecs create-service \
    --cluster financial-ai-cluster \
    --service-name financial-ai-loan-service \
    --task-definition financial-ai-loan-prediction \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers targetGroupArn=$LOAN_TG_ARN,containerName=loan-prediction-dashboard,containerPort=8502
```

## Step 7: Access Your Applications

After deployment completes:

```bash
# Get ALB DNS name
aws elbv2 describe-load-balancers \
    --names financial-ai-alb \
    --query "LoadBalancers[0].DNSName" \
    --output text
```

Access your applications:
- **Main Platform**: `http://ALB-DNS-NAME:8501`
- **Loan Prediction**: `http://ALB-DNS-NAME:8502`

## Cost Optimization

### Use Fargate Spot
Modify task definitions to use Fargate Spot for 70% cost savings:

```json
"capacityProviderStrategy": [
  {
    "capacityProvider": "FARGATE_SPOT",
    "weight": 1,
    "base": 0
  }
]
```

### Auto Scaling
Set up auto-scaling based on CPU/memory:

```bash
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/financial-ai-cluster/financial-ai-main-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 1 \
    --max-capacity 5
```

## Monitoring

### CloudWatch Logs
Logs are automatically sent to CloudWatch:
- `/ecs/financial-ai-main`
- `/ecs/financial-ai-loan`

### CloudWatch Alarms
Create alarms for monitoring:

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name financial-ai-high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2
```

## Security Best Practices

1. **Use HTTPS**: Set up SSL/TLS certificate with ACM
2. **Secrets Management**: Use AWS Secrets Manager for sensitive data
3. **Network Isolation**: Use private subnets with NAT gateway
4. **WAF**: Enable AWS WAF for additional protection
5. **IAM Policies**: Follow principle of least privilege

## Troubleshooting

### Check service status
```bash
aws ecs describe-services \
    --cluster financial-ai-cluster \
    --services financial-ai-main-service financial-ai-loan-service
```

### View logs
```bash
aws logs tail /ecs/financial-ai-main --follow
aws logs tail /ecs/financial-ai-loan --follow
```

### Check task health
```bash
aws ecs list-tasks --cluster financial-ai-cluster
aws ecs describe-tasks --cluster financial-ai-cluster --tasks TASK_ARN
```

## Estimated Costs (Monthly)

- **ECS Fargate (Main)**: ~$35/month (1 task, 2 vCPU, 4GB)
- **ECS Fargate (Loan)**: ~$18/month (1 task, 1 vCPU, 2GB)
- **Application Load Balancer**: ~$20/month
- **EFS Storage**: ~$0.30/GB/month (~$5 for 17GB)
- **Data Transfer**: Variable based on usage
- **CloudWatch Logs**: ~$0.50/GB

**Total**: ~$80-100/month for basic deployment

## Next Steps

1. Set up custom domain with Route 53
2. Enable SSL with AWS Certificate Manager
3. Implement authentication (Cognito or SSO)
4. Set up CI/CD pipeline with AWS CodePipeline
5. Configure automated backups
6. Implement monitoring and alerting

## Support

For issues or questions:
- Check CloudWatch logs
- Review ECS service events
- Verify security group rules
- Ensure EFS is mounted correctly
