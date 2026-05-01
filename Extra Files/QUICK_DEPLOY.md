# 🚀 Quick Deployment Reference

## One-Command Deployment

```bash
# Complete AWS deployment (after initial setup)
./setup-aws-resources.sh && ./deploy-to-aws.sh
```

## Essential Commands

### Local Testing
```bash
# Test with Docker Compose
docker-compose up -d                    # Start all services
docker-compose logs -f                  # View logs
docker-compose down                     # Stop all services

# Test individual containers
docker build -t financial-ai-main .
docker run -p 8501:8501 -v $(pwd)/data:/app/data financial-ai-main
```

### AWS Deployment
```bash
# Initial setup (one-time)
./setup-aws-resources.sh

# Deploy/Update applications
./deploy-to-aws.sh

# Check deployment status
aws ecs describe-services \
  --cluster financial-ai-cluster \
  --services financial-ai-main-service financial-ai-loan-service
```

### Monitoring
```bash
# View CloudWatch logs
aws logs tail /ecs/financial-ai-main --follow
aws logs tail /ecs/financial-ai-loan --follow

# Check container health
docker ps                               # Local
aws ecs list-tasks --cluster financial-ai-cluster  # AWS
```

## Application URLs

**Local:**
- Main Platform: http://localhost:8501
- Loan Prediction: http://localhost:8502

**AWS (after deployment):**
- Get ALB DNS: `aws elbv2 describe-load-balancers --names financial-ai-alb`
- Main Platform: http://ALB-DNS:8501
- Loan Prediction: http://ALB-DNS:8502

## Files You Need to Edit

### Before First Deployment
1. **`.env`** (copy from `.env.example`)
   - Set `AWS_ACCOUNT_ID`
   - Set `AWS_REGION`

2. **`aws-ecs-task-main.json`**
   - Update `YOUR_ACCOUNT_ID`
   - Update `fs-XXXXXXXXX` (after creating EFS)

3. **`aws-ecs-task-loan.json`**
   - Update `YOUR_ACCOUNT_ID`
   - Update `fs-XXXXXXXXX` (after creating EFS)

## Troubleshooting

```bash
# Check Docker
docker --version
docker ps

# Check AWS CLI
aws --version
aws sts get-caller-identity

# Check application health
curl http://localhost:8501/_stcore/health
curl http://localhost:8502/_stcore/health

# Reset everything
docker-compose down -v
docker system prune -a
```

## Cost Estimate

- **Development (local)**: Free
- **AWS (production)**: ~$80-100/month
  - Main Platform: $35/month
  - Loan Prediction: $18/month
  - Load Balancer: $20/month
  - Storage: $5/month
  - Logs: $1/month

## Quick Links

- AWS Console: https://console.aws.amazon.com/
- ECR Repositories: https://console.aws.amazon.com/ecr/
- ECS Clusters: https://console.aws.amazon.com/ecs/
- CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/

## Support

For detailed instructions, see:
- `DEPLOYMENT_CHECKLIST.md` - Complete checklist
- `AWS_DEPLOYMENT_GUIDE.md` - Step-by-step guide
- `DOCKER_GUIDE.md` - Docker reference
