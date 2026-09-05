# RiskGraph

### AI-Powered Merchant Risk & Abuse Intelligence Platform

RiskGraph is an AI-powered merchant risk engine designed to detect **refund abuse, promotional abuse, and coordinated abuse patterns** before they result in significant financial loss.

Instead of evaluating customers in isolation, RiskGraph combines **temporal behavior, behavioral signals, graph relationships, anomaly detection, and supervised machine learning** to identify risky accounts and coordinated patterns.

The system goes beyond a simple risk score by estimating **expected financial exposure** and recommending a proportionate, cost-aware intervention.

---

## 🚨 Problem

Merchant abuse is rarely limited to a single suspicious transaction.

Abusers can repeatedly exploit:

- Refund policies
- Promotional offers and coupons
- Multiple accounts
- Shared devices
- Networks and infrastructure
- Addresses
- Payment methods
- Coordinated account relationships

At the same time, shared infrastructure does **not** automatically imply fraud. Legitimate customers such as families or roommates may naturally share devices, networks, or addresses.

RiskGraph therefore treats relationships as **risk signals rather than proof of abuse**, combining them with temporal and behavioral evidence.

The goal is to answer three questions:

> **Who is risky? Why are they risky? What should the merchant do about it?**

---

## 💡 Solution

RiskGraph provides a complete merchant risk investigation workflow:

1. Detect suspicious behavioral patterns
2. Identify relationships between customers and shared infrastructure
3. Detect anomalous behavior
4. Estimate abuse probability
5. Estimate expected financial exposure
6. Recommend a cost-aware intervention
7. Provide graph and model-level explanations
8. Allow analysts to investigate individual customers and their networks

### Risk Decision

Depending on risk and expected exposure, RiskGraph can recommend:

- **ALLOW**
- **ALLOW WITH MONITORING**
- **SOFT REVIEW**
- **STEP-UP VERIFICATION**
- **MANUAL REVIEW**

This allows merchants to reduce potential losses without unnecessarily blocking legitimate customers.

---

# 🧠 Architecture

```text
Raw Transaction & Relationship Data
                │
                ▼
       Temporal Features
                │
                ▼
      Behavioral Features
                │
                ▼
       NetworkX Graph
                │
        ┌───────┴────────┐
        ▼                ▼
 Network Features    GraphSAGE
                         │
                         ▼
                Graph Embeddings
        └────────┬────────┘
                 │
                 ▼
          Feature Fusion
                 │
        ┌────────┴────────┐
        ▼                 ▼
 Isolation Forest      XGBoost
        │                 │
        └────────┬────────┘
                 ▼
      Probability Calibration
                 │
                 ▼
          Expected Loss
                 │
                 ▼
          Risk Decision
                 │
        ┌────────┴────────┐
        ▼                 ▼
    Intervention       SHAP
                         │
                         ▼
                    Explanation
````

---

# 🔬 Machine Learning Pipeline

### 1. Temporal Features

RiskGraph uses point-in-time feature construction so that predictions only use information available up to the prediction timestamp.

This helps prevent future information from leaking into historical predictions.

The evaluation follows a temporal split:

| Dataset    | Period                   |
| ---------- | ------------------------ |
| Training   | January – August 2025    |
| Validation | September – October 2025 |
| Testing    | November – December 2025 |

---

### 2. Graph-Based Intelligence

A heterogeneous relationship graph connects customers with:

* Devices
* Networks
* Addresses
* Payment methods
* Other customers

NetworkX is used to derive structural features such as:

* Degree
* Weighted degree
* Clustering
* Connected component size
* PageRank

GraphSAGE is then used to learn customer representations from the relationship graph.

The final GraphSAGE customer embedding has **32 dimensions**.

---

### 3. Anomaly Detection

An **Isolation Forest** is used to identify unusual behavioral patterns that may not be captured by supervised classification alone.

The anomaly score is included as an additional signal in the final model.

---

### 4. Supervised Risk Model

The fused feature set is passed into an **XGBoost classifier**.

The final pipeline contains **97 features**, combining temporal, behavioral, graph, embedding, and anomaly signals.

Because abuse cases are highly imbalanced, the training process accounts for class imbalance rather than relying on raw accuracy.

---

### 5. Probability Calibration

The raw XGBoost output is probability-calibrated before being used for risk decisions.

This allows the system to use the resulting probability more meaningfully when calculating expected exposure and selecting interventions.

---

### 6. Expected Loss

RiskGraph connects model predictions to merchant economics.

The system estimates:

```text
Expected Loss =
Abuse Probability × Historical Exposure
```

Historical exposure includes relevant refund amounts and promotional discounts observed up to the prediction point.

This is a **risk-weighted exposure estimate**, not a guarantee of loss avoided.

---

### 7. Explainability

RiskGraph uses **SHAP** to identify the features contributing to model predictions.

This helps analysts understand the reasoning behind a risk score instead of treating the model as a complete black box.

---

# 📊 Dataset

RiskGraph uses a synthetic merchant ecosystem generated specifically for this project.

The dataset contains:

| Entity          |   Count |
| --------------- | ------: |
| Customers       |  50,000 |
| Devices         |  20,000 |
| Networks        |  10,000 |
| Addresses       |  30,000 |
| Payment Methods |  40,000 |
| Merchants       |     100 |
| Products        |   5,000 |
| Orders          | 481,779 |
| Refunds         |  51,535 |
| Coupon Events   |  94,895 |
| Relationships   | 309,475 |

The final labeled population contains:

* **48,918 legitimate customers**
* **1,082 abusive customers**
* **35 coordinated abuse rings**

Abuse patterns include:

* Refund abuse
* Promotional abuse
* Coordinated abuse rings
* Disguised coordinated abuse

The dataset also contains legitimate shared-behavior cases to avoid treating every shared device, network, or address as malicious.

---

# 🖥️ Web Application

RiskGraph includes a merchant-facing web application for investigating and managing risk.

### Overview

Provides a high-level view of:

* Customers monitored
* High-risk population
* Expected exposure
* Priority accounts
* Recommended interventions

### Risk Queue

Allows analysts to:

* Search customers
* Filter by risk band
* Filter by recommended action
* Prioritize high-risk accounts
* Review probability and expected exposure

### Customer Investigation

Provides detailed customer-level risk information and supporting behavioral evidence.

### Network Investigation

Visualizes the customer's surrounding network, including:

* Customers
* Devices
* Networks
* Addresses
* Payment methods
* Relationships

The graph supports investigation through node selection, zooming, fitting, and centering.

### Analytics

Provides aggregate views of:

* Risk-band distribution
* Expected exposure
* High-risk population
* Recommended interventions

### Model & System

Exposes the deployed model architecture and configuration, including:

* Feature count
* GraphSAGE embedding size
* Decision threshold
* Cost assumptions
* Temporal evaluation setup
* ML pipeline components

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* XGBoost
* PyTorch
* PyTorch Geometric
* GraphSAGE
* Scikit-learn
* Isolation Forest
* SHAP
* NetworkX

### Backend

* FastAPI
* Python
* REST APIs

### Frontend

* React
* TypeScript
* Vite
* Tailwind CSS

### Data & Infrastructure

* Parquet
* Pandas
* NumPy
* PostgreSQL
* Docker
* Git / GitHub

---

# 🔐 Design Principles

### Point-in-Time Risk

Features and relationships are constructed using information available at the prediction timestamp.

### Shared Infrastructure ≠ Fraud

A shared device, network, address, or payment method is treated as evidence, not proof of abuse.

### Risk ≠ Abuse

The system produces a risk estimate. A high-risk account is not automatically declared fraudulent.

### Cost-Aware Intervention

The goal is not to block as many customers as possible. The system considers potential financial exposure when recommending interventions.

### Explainability

Risk decisions should be supported by understandable behavioral and network evidence.

---

# ⚙️ Project Structure

```text
RiskGraph/
│
├── backend/
│   └── app/
│       ├── main.py
│       └── ml/
│           ├── inference.py
│           └── graphsage.py
│
├── frontend/
│   └── src/
│       └── ...
│
├── model/
│   └── trained model artifacts
│
├── notebooks/
│   └── training.ipynb
│
├── data/
│   └── synthetic datasets
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# 🚀 Running the Project

## Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The web application will then be available through the Vite development server.

---

# 📌 Current Model Configuration

| Configuration              |  Value |
| -------------------------- | -----: |
| Feature Count              |     97 |
| GraphSAGE Embedding        |     32 |
| Decision Threshold         |   0.05 |
| False Positive Cost        |    ₹20 |
| False Negative Cost        | ₹1,000 |
| GraphSAGE Layers           |      2 |
| GraphSAGE Hidden Dimension |     64 |
| XGBoost Estimators         |    800 |

---

# 🎯 Key Takeaway

RiskGraph is built around a simple idea:

> **Merchant risk is not just about what a customer does. It's also about how their behavior connects to other customers, how that behavior evolves over time, and how much financial exposure it creates.**

By combining **temporal intelligence, behavioral modeling, graph analytics, anomaly detection, supervised ML, financial exposure, and explainability**, RiskGraph turns raw merchant activity into actionable risk intelligence.

---

## 👨‍💻 Built By

**Yash Pandey**

A project exploring how graph-based machine learning and cost-aware decision systems can be applied to real-world merchant risk and abuse detection.

```
```
