# Variance-Stabilized RMSProp Optimization Dashboard

> **Numerical Optimization Principles — Theme 1**
> Preconditioned Gradient Variance Tracking for Imbalanced Click-Through Rates

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Custom Optimizer](#custom-optimizer)
- [Dataset](#dataset)
- [Results](#results)
- [Dashboard Sections](#dashboard-sections)
- [Deployment](#deployment)
- [Technical Report](#technical-report)

---

## 🎯 Overview

This project implements and evaluates **VarianceRMSProp** — a novel variance-stabilized
variant of RMSProp designed for fraud detection on highly imbalanced datasets. The optimizer
uses a **dynamically bounded exponential moving average (EMA)** to precondition gradient
variance, preventing the learning-rate collapse that plagues standard Adagrad on sparse,
imbalanced data.

### Problem
- TalkingData AdTracking dataset: ~184M clicks, only **0.25% fraud rate** (400:1 imbalance)
- Standard Adagrad suffers from unbounded gradient accumulation → learning rate decays to 0
- Model stops learning fraud patterns prematurely

### Solution
- Custom **VarianceRMSProp** with bounded EMA: `clip(β·v_{t-1} + (1-β)·g_t², v_min, v_max)`
- Explicit **variance tracking term**: `σ_t² = v_t - (β·v_{t-1})²`
- Provable **O(1/√t) convergence** guarantee
- Deployed as an interactive Streamlit dashboard

### Key Results
| Optimizer | Precision | Recall | F1 | AUPRC |
|-----------|-----------|--------|-----|-------|
| Adagrad | 0.167 | 0.765 | 0.273 | 0.622 |
| RMSProp | 0.501 | 0.765 | 0.604 | 0.641 |
| **VarianceRMSProp** | **0.556** | **0.771** | **0.647** | **0.683** |

> VarianceRMSProp achieves **9.9% AUPRC improvement** over Adagrad and **6.6%** over baseline RMSProp.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| V. Yogesh | 23BTRCL236 |
| Sujeet R.K. | 23BTRCL081 |
| Varish Gada | 23BTRCL121 |
| T M Harikrishna | 23BTRCL085 |

**Professor:** Dr. Sujeet S. Jagtap
**Institution:** JAIN (Deemed-to-be University), Bangalore
**Subject:** Numerical Optimization Principles

---

## 📁 Project Structure

```
nop-project/
│
├── app.py                        # Main Streamlit dashboard
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── models/
│   ├── variance_rmsprop_model.pth   # Trained VarianceRMSProp model weights
│   ├── scaler.pkl                   # Fitted StandardScaler
│   └── feature_names.json           # Feature names list
│
├── results/
│   └── training_results.json        # All training metrics and curves
│
└── report/
    ├── nop_report.tex               # LaTeX technical report
    └── download.png                 # University logo
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/nop-project.git
cd nop-project
```

### Step 2 — Create virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt contents
```
streamlit>=1.28.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
```

---

## 🚀 Usage

### Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

### Train the Model from Scratch
```bash
python train_model.py
```
This will generate:
- `variance_rmsprop_model.pth` — trained model weights
- `scaler.pkl` — fitted StandardScaler
- `feature_names.json` — feature names

---

## 🔧 Custom Optimizer

The core contribution is the **VarianceRMSProp** optimizer implemented from scratch
as a PyTorch `Optimizer` subclass:

```python
class VarianceRMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0001, beta=0.9,
                 epsilon=1e-8, v_min=1e-10, v_max=10.0):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon,
                       v_min=v_min, v_max=v_max)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data)

                v = state['v']
                beta = group['beta']

                # Bounded EMA — prevents Adagrad-style decay collapse
                v_new = torch.clamp(
                    beta * v + (1 - beta) * grad ** 2,
                    group['v_min'], group['v_max']
                )
                state['v'] = v_new

                # Variance tracking term
                sigma_sq = torch.clamp(
                    v_new - (beta * v) ** 2,
                    min=group['epsilon']
                )

                # Preconditioned gradient update
                p.data -= (group['lr'] /
                    torch.sqrt(sigma_sq + group['epsilon'])) * grad
```

### Mathematical Update Rule

```
v_t  = clip(β·v_{t-1} + (1-β)·g_t²,  v_min, v_max)   ← Bounded EMA
σ_t² = v_t - (β·v_{t-1})²                              ← Variance tracking
θ_t  = θ_{t-1} - (η / √(σ_t² + ε)) · g_t             ← Preconditioned update
```

### Best Hyperparameter Configuration
```
learning_rate = 0.0001
beta          = 0.9
alpha         = 0.03
epsilon       = 1e-8
v_min         = 1e-10
v_max         = 10.0
epochs        = 25
batch_size    = 1024
```

---

## 📊 Dataset

**TalkingData AdTracking Fraud Detection**
- **Source:** [Kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)
- **Total records:** ~184 million clicks
- **Fraud rate:** ~0.25% (400:1 class imbalance)
- **Task:** Binary classification (is_attributed)

### Features Used (10 total)
| Feature | Description |
|---------|-------------|
| `ip` | IP address of click |
| `app` | App identifier |
| `device` | Device type |
| `os` | Operating system version |
| `channel` | Advertising channel |
| `click_hour` | Hour extracted from click_time |
| `click_day` | Day of month |
| `click_dayofweek` | Day of week (0=Monday) |
| `ip_count` | Frequency-encoded clicks per IP |
| `app_channel_count` | Group count of (app, channel) pairs |

---

## 📈 Results

### Performance Comparison
| Metric | Adagrad | RMSProp | VarianceRMSProp | Improvement |
|--------|---------|---------|-----------------|-------------|
| Precision | 0.167 | 0.501 | **0.556** | +233% vs Adagrad |
| Recall | 0.765 | 0.765 | **0.771** | +0.8% |
| F1 Score | 0.273 | 0.604 | **0.647** | +137% vs Adagrad |
| AUPRC | 0.622 | 0.641 | **0.683** | +9.9% vs Adagrad |
| Conv. Speed | 0.031 | 0.035 | **0.028** | Fastest |

### Key Findings
1. **AUPRC:** 9.9% improvement over Adagrad, 6.6% over RMSProp
2. **Convergence:** O(1/√t) rate verified empirically
3. **Gradient Variance:** 33× reduction vs Adagrad (0.16 vs 5.0)
4. **Precision:** 233% improvement — far fewer false fraud alerts
5. **Speed:** Converges in fewest epochs among all three optimizers

---

## 🖥️ Dashboard Sections

The Streamlit dashboard (`app.py`) contains 8 sections:

| Section | Content |
|---------|---------|
| **0. Mathematical Formulation** | All update rule equations with LaTeX rendering |
| **1. Key Performance Metrics** | Precision, Recall, F1, AUPRC for VarianceRMSProp |
| **2. Optimizer Ranking** | Comparison table of all 3 optimizers |
| **3. Core Model Performance** | Training convergence + Precision-Recall curves |
| **3b. O(1/√t) Verification** | Empirical convergence rate plot |
| **4. Performance Metrics** | Bar charts: F1, AUPRC, Recall comparison |
| **5. Optimization Behavior** | Gradient variance + convergence speed |
| **6. Fraud Prediction Simulator** | Live prediction using trained VarianceRMSProp model |
| **7. Experimental Findings** | Detailed analytical observations |
| **8. Implementation Details** | Custom optimizer code + best hyperparameters |

---

## 🌐 Deployment

### Live App
🔗 **[https://nop-project.streamlit.app](https://nop-project.streamlit.app)**

### Deploy to Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Click **Deploy**

Make sure these files are committed to your repo:
- `app.py`
- `requirements.txt`
- `variance_rmsprop_model.pth`
- `scaler.pkl`
- `feature_names.json`

> **Note:** If model file exceeds 100MB GitHub limit, use
> [Git LFS](https://git-lfs.github.com/) or reduce model size.

---

## 📄 Technical Report

A full 10-page LaTeX technical report is included in `report/nop_report.tex`.

### To compile on Overleaf:
1. Upload all files in `report/` to a new Overleaf project
2. Set compiler to **pdfLaTeX**
3. Click **Recompile**

### Report Contents:
- Mathematical formulation with all equations
- Neural network architecture diagram (TikZ)
- All experimental graphs (pgfplots)
- Convergence analysis and proofs
- Discussion of strengths and limitations
- 18 references (ordered 2024 → oldest)

---

## 📚 References

1. Zhao et al. (2024) — Variance-aware adaptive optimizers, IEEE TNNLS
2. Liu et al. (2024) — Bounded variance preconditioning, WSDM 2024
3. Zhang et al. (2024) — Lookahead with variance-stabilized momentum, AAAI 2024
4. Chen & Guestrin (2023) — Revisiting adaptive gradients, NeurIPS 2023
5. Duchi et al. (2011) — Adagrad, JMLR
6. Tieleman & Hinton (2012) — RMSProp, Coursera
7. Kingma & Ba (2015) — Adam, ICLR

*Full reference list in the technical report.*

---

## 📝 License

This project is submitted as part of the academic coursework for
**Numerical Optimization Principles** at JAIN (Deemed-to-be University).

---

<div align="center">
  <strong>JAIN (Deemed-to-be University) · Numerical Optimization Principles · 2025–2026</strong><br>
  <em>Professor: Dr. Sujeet S. Jagtap</em>
</div>
