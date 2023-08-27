# Multiclass Classification 

## Description

Multiclass classification machine learning model to identify if a fatal motor vehicle incident was due to drunk driving, speeding or other

## Methodology

Exploratory Data Analysis (EDA) was employed to meticulously determine the most suitable features for utilization in the analysis. Features demonstrating a singular or extremely dominant mode were excluded, as they lacked significant variability. To further refine the feature selection process, a secondary assessment was conducted to gauge the impact of each feature, employing Chi-Squared testing.

During the feature processing phase, several enhancements were implemented. This involved the discretization of continuous variables such as driver age and hour of the day, and deaths facilitating a more granular representation of these attributes.

## Modeling

Multiple classification models underwent analysis, encompassing linear regression, random forest, Support Vector Classifier (SVC), K-Nearest Neighbors, Gradient Boosting, and Multi-Layer Perceptron (MLP). Through a comprehensive evaluation involving metrics such as accuracy and Macro-Averaged F1 Score, followed by meticulous hyperparameter tuning, the optimal model emerged.

Subsequently, a Gradient Boosting model was developed.

