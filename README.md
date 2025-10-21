

#  Premier League Clustering Analysis: K-means vs AGNES

This project performs an in-depth clustering analysis of **Premier League football teams** based on their season performance metrics such as goals, wins, draws, and losses.  
The analysis compares **K-means** and **AGNES (Agglomerative Hierarchical Clustering)** to evaluate which method better captures similarities in team performance.

---

## ⚙️ Project Overview

This notebook automates the **entire clustering workflow**, including:
- ✅ Data loading and cleaning (with encoding detection)
- ⚽ Feature engineering to compute team-level statistics
- 📊 Clustering using **K-means** and **AGNES**
- 📈 Evaluation with **Silhouette Score**, **ARI**, and **NMI**
- 🌿 Hierarchical clustering visualization via **dendrograms**
- 🧩 PCA-based 2D cluster visualization
- 📋 Cluster interpretation and summary statistics

---

## 📂 Dataset Information



| Column Name | Description |
|--------------|--------------|
| `HomeTeam` | Home team name |
| `AwayTeam` | Away team name |
| `FTHG` | Full-time home goals |
| `FTAG` | Full-time away goals |
| `FTR` | Full-time result (`H`, `A`, `D`) |




---

## 🚀 Features Engineered

For each team, the script computes:
- `Goals_Per_Game`
- `Goals_Conceded_Per_Game`
- `Goal_Difference`
- `Win_Rate`
- `Total_Wins`
- `Total_Draws`
- `Total_Losses`

These metrics are used as inputs for clustering.

---

## 🧮 Clustering Methods Compared

### 🔹 **K-means Clustering**
- Partitions teams into *k* clusters by minimizing within-cluster variance.
- Optimized using **Silhouette Score**.

### 🔹 **AGNES (Agglomerative Hierarchical Clustering)**
- Builds a hierarchy of clusters using **Ward’s linkage**.
- Visualized via a **dendrogram** for interpretability.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Silhouette Score** | Measures cluster cohesion and separation |
| **Adjusted Rand Index (ARI)** | Compares clustering consistency |
| **Normalized Mutual Information (NMI)** | Measures shared information between clusterings |

---

## 🔍 Key Outputs

- **Silhouette Score Plot**: Optimal number of clusters for K-means
- **Dendrogram**: Hierarchical structure of team clusters
- **PCA Visualization**: 2D comparison of clusters from both methods
- **Cluster Profiles**: Average performance metrics within each cluster
- **Comparison Summary**: Determines the better clustering method

---

## 📈 Example Output Summary

| Metric | K-means | AGNES | Better Method |
|---------|----------|--------|----------------|
| Silhouette Score | 0.47 | 0.43 | K-means |

**Cluster Agreement:**
- Adjusted Rand Index (ARI): `0.62`  
- Normalized Mutual Information (NMI): `0.70`

---



## 🧠 Technologies Used

- **Python 3**
- **Pandas, NumPy** — Data handling
- **Matplotlib, Seaborn, Plotly** — Visualization
- **Scikit-learn** — Clustering & metrics
- **SciPy** — Hierarchical clustering
- **Google Colab** — Interactive environment

---

## 🧰 Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/premier-league-clustering-analysis-kmeans-vs-agnes.git
   cd premier-league-clustering-analysis-kmeans-vs-agnes


