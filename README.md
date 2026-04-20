# 🧠 Customer Segmentation using Unsupervised Learning

## 📌 Overview

This project applies **unsupervised machine learning techniques** to segment customers of a wholesale distributor based on their annual spending behavior across multiple product categories.

The goal is to identify distinct customer groups and provide actionable insights that can help optimize business strategies such as delivery scheduling, marketing, and customer targeting.

---

## 📊 Dataset

The dataset used in this project is the **Wholesale Customers Dataset** from the UCI Machine Learning Repository.

### Features:

* Fresh
* Milk
* Grocery
* Frozen
* Detergents_Paper
* Delicatessen

> Note: Features like `Region` and `Channel` were excluded during clustering to focus purely on spending behavior.

---

## ⚙️ Project Workflow

### 1. Data Exploration

* Statistical analysis of features
* Identification of customer patterns
* Smart sampling (low, medium, high spenders)

### 2. Feature Relevance

* Used **Decision Tree Regressor**
* Evaluated feature importance using **R² score**
* Identified weakly predictable (important) features

---

### 3. Data Preprocessing

* Applied:

  * Log Transformation
  * Box-Cox Transformation (preferred)
* Removed outliers using:

  * **Tukey's Method (IQR)**

---

### 4. Dimensionality Reduction (PCA)

* Reduced features into principal components
* Achieved:

  * **~94% variance with 2 components**
  * **~99% variance with 4 components**

---

### 5. Clustering

Used:

* ✅ **Gaussian Mixture Model (GMM)**

Why GMM?

* Handles overlapping clusters
* Supports soft clustering (probability-based)
* More flexible than K-Means

---

### 📈 Silhouette Scores

| Clusters | Score    |
| -------- | -------- |
| 2        | 0.4219 ✅ |
| 3        | 0.3661   |
| 4        | 0.3112   |
| 5        | 0.2520   |

➡️ Optimal number of clusters: **2**

---

## 🧩 Customer Segments

### 🟢 Segment 0 — Horeca (Hotels / Restaurants / Cafés)

* High spending on:

  * Fresh
  * Frozen
* Low on:

  * Grocery & Detergents
* Represents:

  * Food service businesses

---

### 🔴 Segment 1 — Retailers

* High spending on:

  * Grocery
  * Milk
  * Detergents_Paper
* Represents:

  * Supermarkets & retail stores

---

## 🤖 Interactive Dashboard

A fully interactive dashboard built with **Dash + Plotly**:

### Features:

* 📊 KPI metrics
* 📈 PCA visualization
* 🧠 GMM cluster visualization
* 🔥 Correlation heatmaps
* 📦 Cluster profiling
* 🎯 Customer prediction tool

---

## 🚀 Prediction System

Users can input:

* Fresh
* Milk
* Grocery
* Frozen
* Detergents
* Delicatessen

➡️ The model predicts:

* Customer segment
* Business type (Retail / Restaurant)

---

## 🛠️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* SciPy
* Plotly
* Dash
* Joblib

---

## 📁 Project Structure

```
├── data/
│   └── customers.csv
├── models/
│   ├── pca.pkl
│   ├── gmm.pkl
│   ├── boxcox_lambdas.pkl
│   └── features.pkl
├── utils/
│   ├── load_models.py
│   └── preprocessing.py
├── app.py
├── notebook.ipynb
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```
http://127.0.0.1:8050/
```

---

## 💡 Business Impact

* Optimize delivery schedules
* Personalize marketing strategies
* Improve customer satisfaction
* Enable data-driven decisions

---

## 📌 Future Improvements

* Add more clustering models (DBSCAN, K-Means comparison)
* Deploy dashboard online
* Add real-time data input
* Improve model explainability

---

## ⭐ If you found this project useful, give it a star!
