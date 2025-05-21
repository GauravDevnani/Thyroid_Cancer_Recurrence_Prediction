# 🌟Thyroid_Cancer_Recurrence_Prediction🌟
Thyroid Cancer Recurrence Prediction using Artificial Neural Network (ANN)
Overview
This project aims to predict the recurrence of differentiated thyroid cancer using an Artificial Neural Network (ANN). The notebook covers comprehensive Exploratory Data Analysis (EDA), data preprocessing, ANN model building, training, and evaluation.

Dataset
The dataset used in this project is Thyroid_Diff.csv, sourced from Kaggle. It contains clinical and pathological information about patients with differentiated thyroid cancer.

Source: Differentiated Thyroid Cancer Recurrence Dataset
Shape: The dataset comprises 383 samples (patients) and 17 features.
Features: Includes various patient attributes such as Age, Gender, Smoking history, Thyroid Function, Pathology, Focality, Risk, Stage, and more.
Target Variable: The primary target variable is Recurred (Yes/No), indicating whether the cancer recurred.
Missing Values: Notably, the dataset has no missing values, simplifying the initial data cleaning process.
Exploratory Data Analysis (EDA)
The EDA phase involved understanding the dataset's structure, data types, and initial distributions:

Identified that most features are categorical (object type), except for Age, which is numerical.
Confirmed the absence of null values across all columns.
A deeper EDA would typically involve analyzing the distribution of the target variable (Recurred) and exploring relationships between features and the target to gain insights into factors influencing recurrence.
Data Preprocessing
Before feeding the data into the neural network, the following preprocessing steps were performed:

Categorical Encoding: All categorical features (which constitute the majority of the dataset) are converted into numerical representations suitable for machine learning models. This typically involves one-hot encoding.
Feature Scaling: Numerical features, such as Age, are scaled using StandardScaler to normalize their range, which helps the neural network converge faster and perform more effectively.
Data Splitting: The dataset is split into training and testing sets to evaluate the model's generalization performance on unseen data.
Model Architecture: Artificial Neural Network (ANN)
An Artificial Neural Network (ANN) is constructed using TensorFlow and Keras. The model is a sequential stack of fully connected (Dense) layers, designed for binary classification (recurrence prediction).

The network employs activation functions (e.g., ReLU for hidden layers) and includes a final output layer with a Sigmoid activation function to output the probability of recurrence.
The architecture aims to learn complex non-linear relationships within the patient data to predict the recurrence outcome.
Results
The trained ANN model achieved a high accuracy in predicting thyroid cancer recurrence:

Accuracy: 94.81% on the test set.
Further evaluation metrics such as the Confusion Matrix and Classification Report (Precision, Recall, F1-Score) would provide a more detailed understanding of the model's performance for both recurrent and non-recurrent cases.

Dependencies
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
visualkeras
