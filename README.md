# Wine Quality Classification

## a. Problem Statement:
The goal of this project is to build and evaluate multiple machine learning models to classify wine quality based on physicochemical properties. Wine quality is rated on a scale from 3 to 8, and the task is to predict the correct quality class given the input features.  
This assignment demonstrates an end-to-end ML workflow: model training, evaluation, saving models, building an interactive Streamlit app, and deploying it on Streamlit Community Cloud.

## b. Dataset Description:
- **Source**: UCI Machine Learning Repository – Wine Quality Dataset (Red Wine).  
- **Instances**: 1,599 samples.  
- **Features**: 11 physicochemical properties (numeric continuous values), including:  
  - Fixed acidity  
  - Volatile acidity  
  - Citric acid  
  - Residual sugar  
  - Chlorides  
  - Free sulfur dioxide  
  - Total sulfur dioxide
 
## c. Models used:
| Model               |   Accuracy |   Precision |   Recall |       F1 |      MCC |      AUC |
|:--------------------|-----------:|------------:|---------:|---------:|---------:|---------:|
| Logistic Regression |   0.590625 |    0.569525 | 0.590625 | 0.567298 | 0.32502  | 0.76399  |
| Decision Tree       |   0.609375 |    0.612092 | 0.609375 | 0.609477 | 0.398241 | 0.658352 |
| KNN                 |   0.609375 |    0.584116 | 0.609375 | 0.595887 | 0.373313 | 0.698329 |
| Naive Bayes         |   0.5625   |    0.574461 | 0.5625   | 0.568067 | 0.329911 | 0.683783 |
| Random Forest       |   0.675    |    0.650369 | 0.675    | 0.660332 | 0.476837 | 0.766131 |
| XGBoost             |   0.653125 |    0.648027 | 0.653125 | 0.643372 | 0.445301 | 0.798961 |

## Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Achieved moderate accuracy (0.59) and AUC (0.76). It provided a decent baseline but struggled with capturing non-linear relationships in the dataset. |
| Decision Tree        | Slightly better accuracy (0.61) than Logistic Regression. It captured non-linear patterns but risked overfitting, as seen in the lower AUC (0.65). |
| kNN                  | Similar accuracy (0.61) to Decision Tree, but performance was sensitive to neighborhood size. Moderate F1 (0.59) and AUC (0.69) show it was less robust compared to ensemble methods. |
| Naive Bayes          | Lowest accuracy (0.56) among all models. Assumptions of feature independence did not hold well for this dataset, leading to weaker performance despite a reasonable AUC (0.68). |
| Random Forest (Ensemble) | Best overall accuracy (0.675) and F1-score (0.66). Strong MCC (0.47) indicates balanced predictions. Ensemble averaging helped reduce overfitting and improved robustness. |
| XGBoost (Ensemble)   | Slightly lower accuracy (0.65) than Random Forest but achieved the **highest AUC (0.80)**, showing excellent class separation. It handled complex feature interactions well and was competitive with Random Forest. |
