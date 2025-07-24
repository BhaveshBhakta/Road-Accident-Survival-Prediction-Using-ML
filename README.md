## Road Accident Survival Prediction

### Project Overview

This project aims to predict the **survival outcome of individuals involved in road accidents** based on factors such as age, gender, speed of impact, helmet usage, and seatbelt usage. The goal is to develop a machine learning model that can assess the likelihood of survival, which could potentially inform safety campaigns, emergency response strategies, and vehicle safety features.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Road Accident Survival Dataset](https://www.kaggle.com/datasets/himelsarder/road-accident-survival-dataset)
  * **Size**: 200 entries, 6 columns
  * **Key Features**:
      * Age, Gender, Speed\_of\_Impact, Helmet\_Used, Seatbelt\_Used
  * **Approach**:
      * Data Cleaning (Filling missing numerical values with the mean, no duplicates found). Missing categorical values in 'Gender' were handled implicitly by Label Encoding, which assigns a numerical value.
      * Exploratory Data Analysis (Histograms, Boxplots, Heatmaps).
      * Label Encoding for all categorical features (including numerical columns which were converted by the encoder).
      * Data Standardization using `StandardScaler`.
      * Handling Class Imbalance with `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data.
      * Binary Classification (Survived: `1`, Not Survived: `0`).
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree
  * **Best Accuracy**:
      * \~57.5% with Logistic Regression and Ridge Classifier.
      * \~55% with Random Forest Classifier and Bagging Classifier.
      * Note: The relatively low accuracy across models suggests challenges with the dataset, potentially due to its small size, feature representation after encoding, or inherent complexity of the survival prediction task with these features.

-----

### Purpose and Applications

  * Potentially inform public safety campaigns on the **importance of helmet and seatbelt usage**.
  * Provide insights into factors influencing survival rates in road accidents.
  * Serve as a preliminary tool for assessing risk in accident scenarios (with further development and validation).
  * Contribute to discussions on vehicle safety design and road infrastructure improvements.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Caffeine-Content-Classification-Using-ML.git
cd Caffeine-Content-Classification-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Improving model performance through advanced hyperparameter tuning and exploring different model architectures.
  * Investigating alternative methods for handling missing categorical data, such as mode imputation or more sophisticated encoding.
  * Exploring the impact of different feature engineering strategies.
  * Conducting deeper analysis into the limitations of the current dataset and potential avenues for improvement (e.g., more features, larger dataset).# Road-Accident-Survival-Prediction-Using-ML
