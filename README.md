# Neurodegenerative Disease Classifier

This repository contains the code for a machine learning project focused on classifying neurodegenerative diseases, specifically Alzheimer's Disease (AD) and Parkinson's Disease (PD), from MRI brain images. The project systematically evaluates various image preprocessing techniques and implements a robust feature extraction and classification pipeline.

## Project Overview

Neurodegenerative diseases are a growing global health concern, and early, accurate diagnosis is crucial for treatment and management. This project aims to provide an automated classification system to assist medical professionals in identifying these conditions faster and more accurately, thereby potentially leading to earlier treatment and improved patient outcomes.

## Methodology

The project involves several key stages:

1.  **Data Preprocessing Evaluation:** A systematic evaluation of 23 different preprocessing techniques was conducted to optimize classification accuracy. Standard Histogram Equalization was identified as the best preprocessing approach, achieving 80% classification accuracy during this phase.

    ![Data before and after applying Histogram Equalization](https://github.com/user-attachments/assets/9ac614d5-5806-4f29-bf3e-d4f86d14c638)
    *Data before and after applying Histogram Equalization*

    ![Overall scores for different preprocessing techniques](https://github.com/user-attachments/assets/cd80f7c0-e14d-4bf1-be71-c16759ae9eba)
    *Overall scores for different preprocessing techniques*

2.  **Feature Extraction:** A comprehensive set of features were extracted from the 2D MRI slices, including:
    * First-Order Statistical Features (e.g., mean, standard deviation, skewness, kurtosis, entropy, percentiles, min, max, range) 
    * Gray Level Co-occurrence Matrix (GLCM) Features (e.g., contrast, dissimilarity, homogeneity, energy, correlation, ASM) 
    * Local Binary Pattern (LBP) Features 
    * Wavelet Features (from 2D Discrete Wavelet Transform using 'dbl' wavelet) 
    * Edge and Gradient Features (using Sobel and Canny operators) 
    * Gray Level Run Length Matrix (GLRLM) Features 

    ![Distribution of Selected Features Across Diagnostic Classes](https://github.com/user-attachments/assets/615d9467-e126-40ca-a90d-ac5769849241)
    *Distribution of Selected Features Across Diagnostic Classes*

3.  **Feature Selection:** Multiple feature selection methods (Fisher Score, Mutual Information, Random Forest Feature Importance, Correlation Coefficient, Lasso Regression, Recursive Feature Elimination (RFE), T-Test, Elastic Net, SVM, Gradient Boosting) were employed to identify the most informative features. The T-Test emerged as the most effective method, yielding the highest cross-validation accuracy of 0.799 with 10 features.

    | Method                          | Best Number of Features | Cross-Validation Score |
    | :------------------------------ | :---------------------- | :--------------------- |
    | Fisher Score (fclassif)         | 30                      | 0.786                  |
    | Mutual Information              | 15                      | 0.796                  |
    | Random Forest                   | 25                      | 0.792                  |
    | Correlation                     | 20                      | 0.794                  |
    | Lasso Regression                | 25                      | 0.790                  |
    | Recursive Feature Elimination (RFE) | 10                      | 0.797                  |
    | T-Test                          | 10                      | 0.799                  |
    | Elastic Net                     | 20                      | 0.798                  |
    | Support Vector Machine (SVM)    | 30                      | 0.797                  |
    | Gradient Boosting               | 30                      | 0.792                  |
    *Table I: Feature Selection Methods and Performance* 

4.  **Classification:** A Support Vector Machine (SVM) model utilizing a One-Vs-Rest strategy for multi-class classification was trained. The RBF kernel function was used to map non-linear data into a higher dimensional space where separation is easier. The SVM classifier achieved an overall accuracy of 87.5% in distinguishing between Alzheimer's, Parkinson's, and normal brain MRI images.

    ![Confusion Matrix](https://github.com/user-attachments/assets/8afccc99-c605-49ae-a0b9-a3e009f6a34c)
    
    *Confusion Matrix*

    ![ROC Curve](https://github.com/user-attachments/assets/bf94808a-a6df-4357-8b03-52057d51d7c3)
    
    *ROC Curve* 

## Dataset

Our dataset consists of brain medical images categorized into three classes: Alzheimer's disease, Parkinson's disease, and normal controls.
* Parkinson's disease: 2391 images 
* Alzheimer's disease: 2500 images 
* Normal controls: 2699 images 

The dataset was divided using stratified sampling with an 80% training set and 20% testing set. A balanced subset of 50 images per class was used for the preprocessing evaluation phase to ensure fair comparison between techniques.

## Results

* **Optimal Preprocessing:** Standard Histogram Equalization.
* **Overall Classification Accuracy:** 87.5%.
* **Class-wise Accuracy:**
    * Alzheimer: 85.20% 
    * Parkinson: 97.49% 
    * Normal: 80.93% 
* **AUC Scores:** Ranged from 0.93 to 1, indicating excellent separability between classes.

Further details on results, including confusion matrices and ROC curves, can be found in the [paper](https://drive.google.com/file/d/1wCyY5RxwrbDUlspvcRBalUtwrGtuFgvS/view?usp=sharing).

## Contributors

* **Kareem Abdel Nabi:** [GitHub Profile](https://github.com/karreemm)
* **Youssef Aboelela:** [GitHub Profile](https://github.com/Youssef-Abo-El-Ela)
* **RawanAhmed444**: [GitHub Profile](https://github.com/RawanAhmed444)
* **Mostafa Ayman:** [GitHub Profile](https://github.com/mostafa-aboelmagd)
