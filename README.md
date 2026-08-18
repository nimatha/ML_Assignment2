# Problem Statement

This project focuses on a binary classification problem: predicting whether a credit card customer will default on their payment in the next month. The objective is to clean the dataset, prepare it for machine learning, compare multiple models, and select the best-performing classifier for default prediction using the UCI "Default of Credit Card Clients" dataset.

The assignment requires a practical machine learning workflow that includes exploratory data analysis, cleaning of invalid category codes, preprocessing, train/test splitting, model training, evaluation using multiple metrics, and deployment of a simple Streamlit-based evaluation interface.

# Dataset Description

The dataset used in this project is the UCI Default of Credit Card Clients dataset. It contains information about customers' demographic attributes, payment history, billing amounts, and repayment behavior. The target variable is the binary label indicating whether the client defaulted on payment in the following month.

To make the data consistent with the original specification, invalid EDUCATION values (0, 5, 6) were mapped to the nearest valid category, and invalid MARRIAGE values (0) were corrected. Duplicate rows were removed to avoid repeated records affecting the model. The final dataset was then split into an 80/20 stratified train/test set for evaluation.

# GitHub Repository Link

Repository URL: https://github.com/nimatha/ML_Assignment2
Live Streamlit App URL: 

# Models Used

The table below compares the five trained classifiers on the held-out test set, using the required metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7767 | 0.7629 | 0.4960 | 0.5656 | 0.5285 | 0.3845 |
| Decision Tree Classifier | 0.7300 | 0.6037 | 0.3870 | 0.3771 | 0.3820 | 0.2093 |
| KNN Classifier | 0.7951 | 0.6950 | 0.5617 | 0.3363 | 0.4208 | 0.3205 |
| Gaussian Naive Bayes | 0.7964 | 0.7337 | 0.6532 | 0.1704 | 0.2703 | 0.2576 |
| Random Forest Classifier | 0.8145 | 0.7631 | 0.6542 | 0.3424 | 0.4495 | 0.3774 |

# How to Run

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline scripts in order:
   ```bash
   python step1_eda_cleaning.py
   python step2_preprocessing_split.py
   python step3_train_models.py
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload the generated `test_data.csv` file and select a saved model for evaluation.

# Observations

| Model | Observation |
|---|---|
| Logistic Regression | Logistic Regression performed competitively because the relationship between repayment behavior and default risk is fairly structured, and the model produces stable probability estimates. It handled the class imbalance reasonably well and achieved a strong balance between recall and AUC, making it one of the better-performing models on this dataset. |
| Decision Tree Classifier | The Decision Tree underperformed because it is sensitive to noisy credit-risk patterns and tends to create overly specific splits that do not generalize well to unseen data. This is especially problematic in a dataset where repayment history variables are highly correlated and the default class is relatively small. |
| KNN Classifier | KNN achieved solid accuracy but lower recall because the default class is difficult to isolate in a feature space where payment behavior is highly variable. It tends to rely on local similarity, which is less effective when the default pattern is not neatly clustered and the class imbalance is pronounced. |
| Gaussian Naive Bayes | Gaussian Naive Bayes showed very high precision but poor recall, suggesting it was conservative when identifying defaulted clients. This is consistent with the model's assumption of independent features, which does not match the real dependency structure among customer payment and billing variables. |
| Random Forest Classifier | Random Forest was the strongest model because it captures nonlinear interactions between payment history, bill amounts, and credit limits while reducing variance through ensemble averaging. Its ability to model complex credit-default relationships led to the highest accuracy and AUC on the held-out test set. |

# Overall Winner

The Random Forest Classifier is the overall winner for this dataset. It achieved the highest Accuracy (0.8145) and the highest AUC Score (0.7631), while also delivering strong Precision (0.6542). This indicates that the model generalizes better to unseen credit default cases because it captures the nonlinear patterns in payment behavior and billing history more effectively than the other classifiers. Although Logistic Regression was competitive, the Random Forest produced the most reliable overall performance for this credit risk prediction task.
