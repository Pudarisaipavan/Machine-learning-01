# 💼 Predicting Income Levels with Stacking Ensemble  

This project demonstrates how to build and train a **stacking ensemble** model using **Logistic Regression**, **Random Forest**, and **XGBoost** to predict income levels from the well-known **Adult dataset**. The project includes data preprocessing, model building, hyperparameter tuning, evaluation, and a set of advanced visualizations to interpret the model’s performance.  

---

## 🎯 **Objective**  
The goal of this project is to develop a high-performing classification model to predict whether an individual's income exceeds $50K per year based on demographic and employment-related attributes. We aim to:  
✅ Clean and preprocess the Adult dataset using encoding and scaling  
✅ Build a **stacking ensemble model** using Logistic Regression, Random Forest, and XGBoost  
✅ Optimize hyperparameters using **GridSearchCV**  
✅ Evaluate model performance using ROC AUC, confusion matrix, and classification reports  
✅ Provide interpretability using SHAP values and other visualization techniques  

---

## 📂 **Dataset Overview**  
**Dataset:** Adult Census Income Dataset (UCI Repository)  
- **Samples:** 48,842  
- **Features:** 15 demographic and occupational attributes:  
    - `age` – Age of the individual  
    - `workclass` – Type of employer (e.g., Private, Self-emp, etc.)  
    - `fnlwgt` – Final weight assigned to each row  
    - `education` – Education level (e.g., Bachelors, Masters)  
    - `educational-num` – Numerical version of education level  
    - `marital-status` – Marital status  
    - `occupation` – Job category  
    - `relationship` – Relationship status within the family  
    - `race` – Racial background  
    - `gender` – Male or Female  
    - `capital-gain` – Capital gains  
    - `capital-loss` – Capital losses  
    - `hours-per-week` – Hours worked per week  
    - `native-country` – Country of origin  
    - `income` – **Target**: `<=50K` or `>50K`  

**Target Classes:**  
- `0` → Income <= $50K  
- `1` → Income > $50K  

---

## 🏗️ **Model Overview**  
### **Model: Stacking Ensemble**  
The model combines predictions from multiple base learners into a single prediction using a meta-learner:  
✅ **Base Learners:**  
- Logistic Regression  
- Random Forest  
- XGBoost  

✅ **Meta-Learner:**  
- Logistic Regression (to aggregate the predictions)  

### **Architecture**  
| Model Type | Hyperparameters | Purpose |
|------------|-----------------|---------|
| **Logistic Regression** | `solver='liblinear'` | Linear model for interpretability |
| **Random Forest** | `n_estimators=100`, `max_depth=None` | Handles non-linear relationships |
| **XGBoost** | `learning_rate=0.1`, `max_depth=5` | Gradient boosting for high accuracy |

**Stacking Strategy:**  
- The base learners’ predictions are used as input for the meta-learner.  
- The meta-learner learns the best combination of base predictions.  

---

## 🔧 **Hyperparameter Tuning**  
We performed hyperparameter tuning using **GridSearchCV** with the following search space:  

| Parameter | Value Range | Purpose |
|-----------|-------------|---------|
| `final_estimator__C` | [0.1, 1.0, 10.0] | Regularization strength for meta-learner |
| `rf__n_estimators` | [100, 150] | Number of trees in the random forest |
| `xgb__max_depth` | [3, 5] | Tree depth in XGBoost |
| `xgb__learning_rate` | [0.05, 0.1] | Learning rate for XGBoost |

## 📊 **Performance Metrics**  
On the test set:  

| **Metric** | **Value** |
|-----------|-----------|
| **Accuracy** | ~88% |
| **Precision (<=50K)** | 90% |
| **Recall (<=50K)** | 94% |
| **Precision (>50K)** | 79% |
| **Recall (>50K)** | 66% |
| **F1-Score** | 72% |
| **ROC AUC Score** | 0.928 |

---

### **Confusion Matrix**  
| **Actual** | **Predicted <=50K** | **Predicted >50K** |
|-----------|---------------------|---------------------|
| **<=50K** | 7014 | 417 |
| **>50K** | 795 | 1543 |

---

## 📈 **Results and Visualizations**  
### ✅ **1. Interactive ROC Curve**  
- **AUC = 0.928** → Strong separation capability  
- The curve rises steeply at the beginning → High specificity and sensitivity  

---

### ✅ **2. Stacked Bar Chart (Confusion Matrix)**  
- Displays how actual vs. predicted classes break down.  
- Large blue bar for **<=50K** predictions → High precision for the majority class.  
- Red section for **False Positives** → The model predicts some higher incomes incorrectly.  

---

### ✅ **3. SHAP Beeswarm Plot**  
**SHAP (SHapley Additive exPlanations)** provides insights into feature importance:  
- **Marital-status_Married-civ-spouse** → Strongest positive impact on predicting >50K.  
- **Age** → Older individuals are more likely to have higher incomes.  
- **Educational-num** → Higher education levels increase the probability of earning >50K.  
- **Capital-gain** → High capital gains strongly push the model toward predicting >50K.  

---

### ✅ **4. Gaussian-Smoothed Learning Curve**  
- Demonstrates consistent model improvement during training.  
- Smooth decline in loss → No signs of overfitting.  

1. **Clone the Repository:**
 ```bash
 git clone https://github.com/Pudarisaipavan/income-classification.git
