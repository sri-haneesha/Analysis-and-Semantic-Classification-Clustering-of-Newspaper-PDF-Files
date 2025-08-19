# Analysis-and-Semantic-Classification-Clustering-of-Newspaper-PDF-Files
Analyzing Text Classification Performance Using Machine Learning Models with BERT Embeddings

**Project Overview**

This project implements an automated newspaper and magazine classification system using BERT embeddings and machine learning models. The system classifies news articles into nine categories:
POLITICS, SPORTS, TECH, ENVIRONMENT, CRIME, BUSINESS, ENTERTAINMENT, COMEDY, WELLNESS.

Five ML models are evaluated for performance:
Support Vector Machines (SVM)
k-Nearest Neighbors (KNN)
Random Forest
Gradient Boosting
Logistic Regression

Evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrices.

**Features**

Preprocessing: Text cleaning, tokenization, stopword removal, lemmatization
Feature Engineering: BERT embeddings for semantic understanding
Model Training: SVM, KNN, Random Forest, Gradient Boosting, Logistic Regression
Evaluation: Model comparison using metrics and visualizations
Optimization: Hyperparameter tuning with GridSearchCV

**Dependencies** 

Python 3.x
pandas
scikit-learn
transformers
tensorflow
matplotlib
seaborn

**Dataset**

Source: Kaggle news dataset
500 records per category
Nine categories as listed above
Preprocessed dataset included 

**Experimental Results**

Random Forest and Gradient Boosting achieved the highest accuracies (~0.65–0.70)
Confusion matrices highlight overlapping categories and classification challenges
BERT embeddings improved semantic understanding and overall performance

**Future Work**

Implement deep learning models (LSTM, CNN, Transformers) for improved accuracy
Explore ensemble learning approaches
Develop a web interface or API for real-time classification
Improve text extraction from PDFs to handle multi-column layouts and graphics

**References**

Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805
Data Extraction from Unstructured PDFs – Analytics Vidhya
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830
TensorFlow Documentation – GPU Support


