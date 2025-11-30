# Unsupervised Learning Projects

This repository contains several unsupervised learning projects that demonstrate clustering, dimensionality reduction, and recommendation techniques on different types of datasets. Each project focuses on exploring patterns in data without labeled outcomes.

---

## 📂 Repository Structure

- `data/` — Contains datasets used in the projects.  
- `ref_imgs/` — Reference images for visualizations (if any).  
- `clustering-pca-k-means-dbscan-hierarchical.ipynb` — Clustering analysis using multiple algorithms.  
- `kmeans_scratch_implementation.ipynb` — K-Means implemented from scratch to understand its internal workings.  
- `news_clustering_unsupervised_ml.ipynb` — Clustering applied to text data (news articles).  
- `collaborative-recommendation.ipynb` — Collaborative recommendation using similarity-based techniques.

---

## 📌 Projects Overview

### 1. Clustering Analysis with PCA, K-Means, DBSCAN, and Hierarchical Clustering
**Notebook:** `clustering-pca-k-means-dbscan-hierarchical.ipynb`  

**Objective:**  
To explore the structure of a dataset by grouping similar entities (e.g., countries) using different clustering techniques.  

**Key Steps:**  
1. Load and preprocess the dataset (handling missing values, scaling features).  
2. Perform exploratory data analysis (EDA) to understand distributions and correlations.  
3. Apply **PCA** (Principal Component Analysis) for dimensionality reduction and visualization.  
4. Apply **K-Means**, **DBSCAN**, and **Hierarchical clustering** to find natural groupings.  
5. Compare clusters across methods and visualize with boxplots, scatter plots, and dendrograms.  

**Learning Outcome:**  
Understand differences between clustering algorithms, how to choose the right one, and how to interpret clusters.

---

### 2. K-Means Implementation from Scratch
**Notebook:** `kmeans_scratch_implementation.ipynb`  

**Objective:**  
To gain a deeper understanding of K-Means by implementing it manually, without using prebuilt libraries.  

**Key Steps:**  
1. Randomly initialize cluster centroids.  
2. Assign each data point to the nearest centroid.  
3. Recalculate centroids based on cluster members.  
4. Repeat until convergence.  
5. Compare results with scikit-learn’s K-Means implementation.  

**Learning Outcome:**  
Learn how K-Means works internally and understand concepts like cluster assignment, centroid calculation, and convergence criteria.

---

### 3. News Clustering Using Unsupervised ML
**Notebook:** `news_clustering_unsupervised_ml.ipynb`  

**Objective:**  
To cluster news articles based on their content without labeled categories.  

**Key Steps:**  
1. Load a dataset of news articles.  
2. Preprocess text: remove punctuation, stopwords, and tokenize.  
3. Convert text into numerical features using vectorization techniques (e.g., TF-IDF).  
4. Apply **K-Means** or other clustering algorithms to group similar articles.  
5. Analyze top words per cluster to interpret topics.  

**Learning Outcome:**  
Learn how unsupervised learning can be applied to textual data and understand text preprocessing and feature extraction.

---

### 4. Collaborative Recommendation (Similarity-Based)
**Notebook:** `collaborative-recommendation.ipynb`  

**Objective:**  
To recommend items (e.g., movies) to users based on similarity with other users or items, using an unsupervised approach.  

**Key Steps:**  
1. Load user-item interaction dataset (e.g., ratings).  
2. Compute similarity between users or items (e.g., cosine similarity).  
3. Identify nearest neighbors and recommend top items.  
4. Compare recommendations with K-Nearest Neighbors approach.  

**Learning Outcome:**  
Understand collaborative filtering concepts and how similarity metrics drive recommendation systems.

---

## 🛠 Tools and Libraries Used
- Python 3  
- Jupyter Notebook  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (for clustering and preprocessing)  
- nltk / sklearn.feature_extraction.text (for text processing)  

---

## 🚀 How to Use This Repository
1. Clone the repository.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
   *(or install packages manually: pandas, numpy, scikit-learn, matplotlib, seaborn, nltk)*  
3. Open the notebooks and run them step by step.  
4. Modify parameters (e.g., number of clusters) to explore different results.  

---

## ℹ️ Notes
- The repository is for learning and experimenting with unsupervised techniques.  
- Each notebook contains explanations, visualizations, and step-by-step procedures.  
- Adding a `requirements.txt` will help replicate the environment easily.

---

**Enjoy exploring unsupervised learning and improving your machine learning skills!**
