# 🎯 MMM Multi-Market Bayesian

**Marketing Mix Modeling for Multi-Market Analysis using Bayesian Inference**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10-red.svg)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🌍 Overview

This project implements a **scalable Marketing Mix Modeling (MMM) solution** designed to analyze marketing effectiveness across **multiple markets** (countries/regions). Inspired by **Google's Meridian** and **Meta's Robyn**, it uses **Bayesian inference** with **PyMC-Marketing** to provide robust, interpretable insights into marketing ROI.

### **Business Context**
Modern enterprises operate across multiple geographic markets, each with unique consumer behaviors, media consumption patterns, seasonal dynamics, and competitive landscapes.

This project demonstrates how to:
- ✅ Build and deploy MMM models at scale (10-100+ markets)
- ✅ Quantify marketing impact with uncertainty estimates
- ✅ Optimize budget allocation across channels and markets
- ✅ Automate model training and monitoring pipelines

---

## ⚡ Key Features

### **Modeling**
- 🎯 **Bayesian MMM** with PyMC-Marketing
- 📊 **Adstock transformations** (geometric, delayed) for carryover effects
- 📈 **Saturation curves** (Hill, logistic) for diminishing returns
- 🧮 **Hierarchical models** for cross-market learning
- 🔍 **MCMC diagnostics** (R-hat, ESS, posterior predictive checks)

### **Scalability**
- 🔄 **Parallel training** across markets
- 📦 **MLflow tracking** for experiment management
- 🚀 **Airflow orchestration** for automated pipelines

### **Interpretability**
- 💰 **Channel-level ROI** with credible intervals
- 📉 **Contribution decomposition** (base sales, marketing, seasonality)
- 🎨 **Response curves** for budget optimization
- 📊 **Cross-market comparisons**

### **Production-Ready**
- 🧪 **Unit tests** with pytest
- 📝 **Comprehensive documentation**
- 🎛️ **Interactive Streamlit dashboard**

---

## 🏗️ Project Architecture

```
mmm-multi-market-bayesian/
├── config/
│   ├── model_config.yaml
│   ├── markets_config.yaml
│   └── priors.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_market_poc.ipynb
│   ├── 03_multi_market_training.ipynb
│   └── 04_model_diagnostics.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── feature_engineering.py
│   │   └── multi_market_generator.py
│   ├── models/
│   │   ├── base_mmm.py
│   │   ├── bayesian_mmm.py
│   │   ├── adstock.py
│   │   └── saturation.py
│   ├── training/
│   │   ├── distributed_trainer.py
│   │   ├── hyperparameter_tuning.py
│   │   └── model_diagnostics.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── model_validation.py
│   │   └── roi_calculator.py
│   └── utils/
│       ├── logging_config.py
│       └── visualization.py
├── pipelines/
│   ├── airflow_dags/
│   └── orchestration/
├── app/
│   ├── streamlit_app.py
│   └── components/
├── tests/
├── docs/
│   ├── methodology.md
│   ├── data_dictionary.md
│   └── api_reference.md
├── results/
│   ├── models/
│   ├── diagnostics/
│   ├── reports/
│   └── visualizations/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/mmm-multi-market-bayesian.git
cd mmm-multi-market-bayesian
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data/multi_market_generator.py
```

---

## ⚡ Quick Start

```bash
# 1. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Train single market (POC)
python src/training/train_single_market.py --market FR

# 3. Train all markets
python src/training/distributed_trainer.py --markets all

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 📊 Data

### **Source: Robyn (Meta/Facebook)**
- **208 weeks** of historical data
- **5 marketing channels**: TV, Out-of-Home, Print, Facebook, Search
- **10 European markets**: FR, DE, UK, IT, ES, NL, BE, PL, SE, NO
- **2 080 lignes** exactes (10 pays × 208 semaines)
- **Validation : 0 erreur** sur 4 niveaux de tests

| Market | GDP Index | Digital Maturity | TV Dominance | Seasonality   |
|--------|-----------|------------------|--------------|---------------|
| FR     | 1.00      | 0.85             | 0.70         | Standard      |
| DE     | 1.15      | 0.90             | 0.65         | Standard      |
| UK     | 1.10      | 0.95             | 0.60         | Mild          |
| IT     | 0.85      | 0.75             | 0.75         | Mediterranean |
| ES     | 0.80      | 0.78             | 0.72         | Mediterranean |
| NL     | 1.25      | 0.98             | 0.50         | Standard      |
| BE     | 1.05      | 0.88             | 0.68         | Standard      |
| PL     | 0.65      | 0.82             | 0.78         | Eastern       |
| SE     | 1.30      | 0.99             | 0.45         | Nordic        |
| NO     | 1.50      | 0.97             | 0.48         | Nordic        |

---

## 🧮 Methodology

```
Sales(t) = BaseSales
         + Σ[β_i × Adstock(Spend_i(t)) × Saturation(Spend_i(t))]
         + Seasonality(t) + Trend(t) + Events(t) + ε(t)
```

See [docs/methodology.md](docs/methodology.md) for full mathematical details.

---

## 📈 Results

*(To be populated after model training)*

| Metric | Market FR | Market DE | Market UK | Average |
|--------|-----------|-----------|-----------|---------|
| R²     | TBD       | TBD       | TBD       | TBD     |
| MAPE   | TBD       | TBD       | TBD       | TBD     |
| NRMSE  | TBD       | TBD       | TBD       | TBD     |

---

## 🎛️ Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

---

## 📜 License

MIT License

---

## 👤 Contact

**Samir EL AISSAOUY** — Data Consultant | Data Engineer / Analyst

- 📧 Elaissaouy.samir12@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/samir-el-aissaouy)
- 📞 +33 7 52 07 68 61
- 📍 Île-de-France, France

---

## 📊 Project Status

🚧 **Active Development**

- [x] Project structure
- [ ] Data generation (multi_market_generator.py)
- [ ] Single market POC
- [ ] Multi-market training
- [ ] Dashboard development
- [ ] Documentation
- [ ] Testing
- [ ] Docker containerization
