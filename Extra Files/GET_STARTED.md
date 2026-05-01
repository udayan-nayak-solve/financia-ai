# 🎯 Getting Started with Financial AI Platform

**Welcome!** This guide will get you up and running in 5 minutes.

## 📋 What You'll Need

- **Docker Desktop** (recommended) OR Python 3.8+
- **4GB RAM** minimum
- **2GB disk space**
- **Data files** in `data/actual/` folder

## 🚀 Option 1: Docker (Fastest - Recommended)

### Step 1: Build
```bash
./docker-run.sh build
```
⏱️ Takes ~5 minutes

### Step 2: Run
```bash
./docker-run.sh up
```

### Step 3: Access
Open your browser to: **http://localhost:8501**

**That's it! ��**

---

## 🐍 Option 2: Python (Manual Setup)

### Step 1: Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train Models
```bash
# Train all models
python src/comprehensive_pipeline.py

# Or train loan models only
cd loan_prediction_system
python src/training_pipeline.py
```

### Step 3: Launch Dashboard
```bash
streamlit run src/executive_dashboard.py
```

### Step 4: Access
Open browser: **http://localhost:8501**

---

## 🎯 What You Can Do

### 1. **Analyze Market Opportunities** (Main Dashboard)
- View census tract opportunity scores
- See geographic maps with performance heat maps
- Review temporal forecasts (2025-2026)
- Analyze year-over-year trends

### 2. **Predict Loan Outcomes** (Loan Dashboard)
```bash
# Open loan prediction dashboard (new terminal)
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502
```
Access: **http://localhost:8502**

- Fill out loan application form
- Get instant approval/denial prediction
- See confidence scores and risk analysis
- View denial reasons (if applicable)

### 3. **Compare Year-over-Year Performance**
```bash
# Open YoY analysis (new terminal)
streamlit run src/enhanced_yoy_dashboard.py --server.port 8503
```
Access: **http://localhost:8503**

- Compare 2022 vs 2023 vs 2024
- See performance rankings
- Identify growth opportunities
- Get strategic insights

---

## 📊 Your First Analysis

### Scenario: Evaluate a Loan Application

1. **Open loan dashboard**: http://localhost:8502

2. **Enter applicant info**:
   - Income: $85,000
   - Credit Score: 740
   - Loan Amount: $350,000
   - Property Value: $450,000

3. **View results**:
   - ✅ Approved with 92.3% confidence
   - Risk score: 28.5 (Low risk)
   - Key factors: Strong credit, healthy DTI ratio

4. **Try adjusting values** to see how predictions change!

---

## 🔄 Common Workflows

### Daily Market Review
```bash
# Open main dashboard
streamlit run src/executive_dashboard.py

# Navigate to:
# • Overview tab → Market summary
# • Temporal Forecasting → Future trends
# • YoY Analysis → Performance tracking
```

### Loan Processing
```bash
# Option A: Use main dashboard → Loan Predictions tab
streamlit run src/executive_dashboard.py

# Option B: Use dedicated loan dashboard
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502
```

### Model Retraining (Weekly/Monthly)
```bash
# Update data files in data/actual/

# Retrain all models
python src/comprehensive_pipeline.py

# Retrain loan models
cd loan_prediction_system
python src/training_pipeline.py

# Restart dashboards to use new models
```

---

## 🆘 Quick Troubleshooting

### Dashboard won't load?
```bash
# Check if port is in use
lsof -i :8501

# Try different port
streamlit run src/executive_dashboard.py --server.port 8505
```

### Models not found?
```bash
# Train the models first
python src/comprehensive_pipeline.py
cd loan_prediction_system && python src/training_pipeline.py
```

### Docker issues?
```bash
# Check logs
docker logs financial-ai-dashboard

# Restart clean
./docker-run.sh clean
./docker-run.sh build
./docker-run.sh up
```

### Import errors?
```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or reinstall
pip install --upgrade -r requirements.txt
```

---

## 📚 Next Steps

Now that you're running, explore these guides:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - All commands at a glance
2. **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Visual architecture and workflows
3. **[DOCKER.md](DOCKER.md)** - Advanced Docker deployment
4. **[README.md](README.md)** - Full documentation

---

## 💡 Pro Tips

### Run all dashboards simultaneously
```bash
# Terminal 1: Main dashboard
streamlit run src/executive_dashboard.py

# Terminal 2: Loan predictions
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502

# Terminal 3: YoY analysis
streamlit run src/enhanced_yoy_dashboard.py --server.port 8503
```

### Use Docker for production
```bash
# Start all services
docker-compose --profile yoy up -d

# Main: http://localhost:8501
# YoY: http://localhost:8502
```

### Quick health check
```bash
# Python approach
python -c "from loan_prediction_system.src.prediction_service import get_prediction_service; print('✅ System ready!' if get_prediction_service().health_check() else '❌ Issue detected')"

# Docker approach
docker ps  # Should show "healthy" status
```

---

## 🎓 Learning Path

### Week 1: Basics
- [ ] Set up and run main dashboard
- [ ] Explore opportunity scores
- [ ] Review geographic maps
- [ ] Make sample loan predictions

### Week 2: Advanced
- [ ] Train your own models
- [ ] Configure model parameters
- [ ] Compare YoY performance
- [ ] Generate strategic insights

### Week 3: Production
- [ ] Deploy with Docker
- [ ] Set up automated retraining
- [ ] Create custom workflows
- [ ] Integrate with your systems

---

## 📞 Need Help?

1. **Check logs**: `tail -f logs/pipeline.log`
2. **Review docs**: See [README.md](README.md)
3. **Quick commands**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. **Visual guide**: See [VISUAL_GUIDE.md](VISUAL_GUIDE.md)

---

**🎉 You're all set! Happy analyzing!**

**Quick Start**: `./docker-run.sh build && ./docker-run.sh up`
