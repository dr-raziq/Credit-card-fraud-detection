CREDIT CARD FRAUD DETECTION

Author: Mohammad Raziq Mohammad Arshad Shaikh

A Machine Learning-Based Fraud Detection System
Final Year B.Sc. Computer Science Project - A+ Grade

Ashoka Center for Business and Computer Studies
Affiliated with Savitribai Phule Pune University (SPPU)
2025

1. PROJECT OVERVIEW
Credit card fraud is a major challenge in the financial sector, where fraudulent transactions can result in significant financial losses for customers and financial institutions.
This project presents a machine learning-based Credit Card Fraud Detection System designed to identify potentially fraudulent transactions from historical transaction data.
The system combines data preprocessing, exploratory data analysis, class-imbalance handling, machine learning model training, hyperparameter tuning, model evaluation, and explainable artificial intelligence.
The project focuses on developing a reliable fraud detection workflow while giving importance to fraud recall, precision, F1-score, ROC-AUC, and PR-AUC, rather than relying only on overall accuracy.

2. PROJECT OBJECTIVES
The main objectives of this project are:
•	To analyze historical credit card transaction data.
•	To identify patterns associated with fraudulent transactions.
•	To preprocess and standardize relevant transaction features.
•	To handle severe class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
•	To train multiple machine learning classification models.
•	To compare model performance using appropriate evaluation metrics.
•	To perform hyperparameter tuning for improved model performance.
•	To use SHAP for model explainability and feature interpretation.
•	To develop an interactive fraud detection interface using Streamlit.
•	To create a complete and reproducible machine learning workflow.

3. DATASET
The project uses a credit card transaction dataset containing anonymized transaction features.
The dataset includes:
•	Time – Time elapsed between transactions.
•	Amount – Transaction amount.
•	V1 to V28 – Anonymized numerical features generated through dimensionality transformation.
•	Class – Target variable indicating whether the transaction is fraudulent.
The target variable is represented as:
•	0 – Legitimate Transaction
•	1 – Fraudulent Transaction
The dataset is highly imbalanced because fraudulent transactions represent only a small proportion of all transactions. Therefore, special techniques are required during model development and evaluation.

4. PROJECT WORKFLOW
The overall workflow of the project is:
Raw Dataset
↓
Data Cleaning and Validation
↓
Exploratory Data Analysis
↓
Feature Preprocessing
↓
Class Imbalance Handling
↓
Train-Test Split
↓
Machine Learning Model Training
↓
Hyperparameter Tuning
↓
Model Evaluation
↓
SHAP Explainability
↓
Streamlit Deployment

5. KEY FEATURES
Data Preprocessing
•	Dataset validation and cleaning.
•	Feature scaling.
•	Train-test splitting.
•	Stratified sampling to preserve class distribution.
Class Imbalance Handling
The dataset contains significantly fewer fraudulent transactions than legitimate transactions.
To address this issue, the project uses SMOTE, which generates synthetic samples for the minority class and helps machine learning models learn fraudulent transaction patterns more effectively.
Machine Learning
Multiple classification algorithms are implemented and compared:
1.	Logistic Regression
2.	Random Forest
3.	XGBoost
Hyperparameter Tuning
Model hyperparameters are optimized to improve classification performance and obtain better generalization.
Model Evaluation
The models are evaluated using:
•	Precision
•	Recall
•	F1-Score
•	ROC-AUC
•	PR-AUC
•	Confusion Matrix
Explainable AI
SHAP (SHapley Additive exPlanations) is used to understand the contribution of individual features to model predictions.
Interactive Application
A Streamlit-based interface is used to provide an interactive way to test transaction data and display fraud detection results.

6. MACHINE LEARNING MODELS
6.1 Logistic Regression
Logistic Regression is used as a baseline classification model.
It estimates the probability that a transaction belongs to the fraudulent class and provides a simple and interpretable reference point for comparing more complex models.
6.2 Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees.
It is capable of modeling nonlinear relationships between transaction features and fraud labels and is generally robust to different feature patterns.
6.3 XGBoost
XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees sequentially.
It is included because of its strong performance on structured and tabular datasets and its ability to capture complex relationships between features.

7. HANDLING CLASS IMBALANCE
One of the major challenges in credit card fraud detection is class imbalance.
A model could achieve a high accuracy by predicting almost every transaction as legitimate while still failing to detect fraudulent transactions.
To reduce this problem, the project applies SMOTE to the training data.
SMOTE generates synthetic minority-class examples based on existing minority samples.
Importantly, the dataset is split into training and testing sets before applying SMOTE so that the test set remains representative of unseen data.

8. FEATURE PREPROCESSING
Feature preprocessing is performed to ensure that numerical variables are suitable for machine learning algorithms.
The project includes feature scaling for selected features such as:
•	Time
•	Amount
Standardization transforms numerical values into a comparable scale.
The preprocessing pipeline also ensures that:
•	The target variable is separated from input features.
•	Training and testing data are separated.
•	Class proportions are preserved through stratified splitting.
•	SMOTE is applied to training data when required.

9. MODEL EVALUATION
Because fraud detection is an imbalanced classification problem, accuracy alone is not sufficient for evaluating the models.
Precision
Precision measures how many transactions predicted as fraudulent are actually fraudulent.
Precision = TP / (TP + FP)
A high precision indicates fewer false fraud alerts.
Recall
Recall measures how many actual fraudulent transactions are successfully detected.
Recall = TP / (TP + FN)
Recall is particularly important in fraud detection because missing fraudulent transactions can have significant consequences.
F1-Score
F1-score is the harmonic mean of precision and recall.
F1 = 2 × (Precision × Recall) / (Precision + Recall)
It provides a balanced measure when both precision and recall are important.
ROC-AUC
ROC-AUC measures the model's ability to distinguish between legitimate and fraudulent transactions across different classification thresholds.
PR-AUC
Precision-Recall AUC is especially useful for imbalanced classification problems because it focuses on the relationship between precision and recall.
Confusion Matrix
The confusion matrix provides four categories:
•	True Positive
•	True Negative
•	False Positive
•	False Negative

10. MODEL RESULTS
The final model performance should be reported using the actual results obtained during experimentation.
Model	 Precision	  Recall	  F1-Score	  ROC-AUC	  PR-AUC
Logistic Regression	0.6098	0.7895	0.6881	0.9614	0.6787
Random Forest	0.8947	0.7158	0.7953	0.971	0.7774
XGBoost	0.9474	0.7579	0.8421	0.9535	0.7976

11. EXPLAINABLE AI WITH SHAP
Machine learning models can sometimes be difficult to interpret, especially when complex ensemble algorithms are used.
To improve transparency, the project uses SHAP (SHapley Additive exPlanations).
SHAP helps explain:
•	Which features contribute most to a prediction.
•	Whether a feature increases or decreases the predicted fraud probability.
•	Why a particular transaction may have been classified as fraudulent.
•	Overall feature importance across the dataset.
This provides greater transparency and helps users better understand machine learning predictions.

12. STREAMLIT APPLICATION
The project includes a Streamlit-based interface for interacting with the trained fraud detection model.
The application can be used to:
    1.	Enter or provide transaction feature values.
    2.	Process the transaction through the trained model.
    3.	Generate a fraud prediction.
    4.	Display the prediction result.
    5.	Present relevant prediction information in an easy-to-understand interface.
The Streamlit application demonstrates how the trained machine learning model can be integrated into a practical user-facing system.

13. PROJECT STRUCTURE
credit-card-fraud-detection/
|
|__ data/
|   |__ raw/
|   |__ processed/
|
|__ notebooks/
|   |__ 01_eda.ipynb
|   |__ 02_preprocessing.ipynb
|   |__ 03_model_training.ipynb
|   |__ 04_model_comparison.ipynb
|   |__ 05_shap_explainability.ipynb
|
|__ src/
|   |__ data/
|   |   |__ preprocess.py
|   |
|   |__ models/
|   |   |__ train.py
|   |   |__ evaluate.py
|   |   |__ tune.py
|   |
|   |__ explainability/
|       |__ shap_analysis.py
|
|__ tests/
|   |__ test_preprocess.py
|
|__ app/
|   |__ app.py
|
|__ models/
|   |__ trained_models/
|
|__ requirements.txt
|__ README.md
|__ LICENSE

14. TECHNOLOGY STACK
Programming Language
•	Python
Data Processing
•	Pandas
•	NumPy
Data Visualization
•	Matplotlib
•	Seaborn
Machine Learning
•	Scikit-learn
•	XGBoost
•	Imbalanced-learn
Explainable AI
•	SHAP
Application Development
•	Streamlit
Testing
•	Pytest
Development Environment
•	Jupyter Notebook
•	Visual Studio Code
•	Git
•	GitHub

15. INSTALLATION
Step 1: Clone the Repository
git clone <repository-url>
cd credit-card-fraud-detection
Step 2: Create a Virtual Environment
python -m venv venv
Step 3: Activate the Virtual Environment
Windows:
venv\Scripts\activate
Linux/macOS:
source venv/bin/activate

Step 4: Place the original credit card dataset from Kaggle at:
data/creditcard.csv

Download from : “https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud”
Step 5: Install Dependencies
pip install -r requirements.txt

Step 6:  Train and Evaluate Model
From the project root:
python3 scripts/train.py --config config/config.yaml

python3 scripts/evaluate.py --config config/config.yaml


16. RUNNING THE PROJECT
Run the Streamlit Application
streamlit run app/app.py
The application will open in a browser and provide an interactive fraud detection interface.
Run Tests
python3 -m pytest
The test suite verifies important preprocessing functionality, including:
•	Feature scaling.
•	SMOTE-based class balancing.
•	Train-test splitting.
•	Stratified class distribution.

17. TESTING
Automated testing is included to verify important components of the preprocessing pipeline.
The current tests cover:
Feature Scaling
Verifies that selected numerical features are properly standardized while unrelated features remain unchanged.
SMOTE
Verifies that minority-class samples are appropriately increased and that the resulting class distribution is balanced for the test case.
Train-Test Split
Verifies:
•	Correct training and testing sizes.
•	Preservation of class distribution.
•	Correct separation of input features and target variable.
Testing helps improve the reliability and maintainability of the project.

18. IMPORTANT CONSIDERATIONS
Fraud detection systems operate in a highly imbalanced environment. Therefore, model evaluation should focus on appropriate metrics rather than accuracy alone.
Particular attention should be given to:
•	False negatives.
•	False positives.
•	Precision.
•	Recall.
•	F1-score.
•	PR-AUC.
•	Generalization to unseen transactions.
The final model should be selected based on the project's experimental results and the specific requirements of the fraud detection application.

19. ACADEMIC CONTEXT
This project was developed as a Final Year B.Sc. Computer Science Project.
Institution:
Ashoka Center for Business and Computer Studies
University Affiliation:
Savitribai Phule Pune University (SPPU)
Year:
2025
Project Grade:
A+
The project demonstrates the practical application of machine learning, data preprocessing, class-imbalance handling, model evaluation, explainable AI, automated testing, and application deployment.

20. CONCLUSION
The Credit Card Fraud Detection project demonstrates an end-to-end machine learning approach for identifying potentially fraudulent credit card transactions.
The project combines:
•	Data preprocessing
•	Exploratory data analysis
•	Class imbalance handling
•	Machine learning classification
•	Hyperparameter tuning
•	Model evaluation
•	Explainable AI
•	Automated testing
•	Streamlit deployment
By combining these components into a single workflow, the project provides a practical demonstration of how machine learning techniques can be applied to a real-world financial fraud detection problem.

21. FUTURE ENHANCEMENTS
Potential future improvements include:
•	Real-time transaction monitoring.
•	Integration with banking APIs.
•	Advanced deep learning models.
•	Online learning for changing fraud patterns.
•	Improved anomaly detection techniques.
•	Automated model retraining.
•	Advanced dashboard and visualization features.
•	Cloud-based deployment.
•	Model monitoring and performance tracking.

22. LICENSE
This project is developed for academic and educational purposes.
Please refer to the LICENSE file in the repository for the applicable license and usage terms.

PROJECT SUMMARY
Project: Credit Card Fraud Detection
Domain: Machine Learning / Financial Fraud Detection
Degree: B.Sc. Computer Science
Institution: Ashoka Center for Business and Computer Studies
University: Savitribai Phule Pune University (SPPU)
Year: 2025
Grade: A+
Core Technologies: Python, Pandas, NumPy, Scikit-learn, XGBoost, SMOTE, SHAP, Streamlit, Pytest
Machine Learning Models: Logistic Regression, Random Forest, XGBoost
Evaluation Metrics: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix
Key Components: Data Preprocessing, EDA, Class Imbalance Handling, Model Training, Hyperparameter Tuning, Explainable AI, Testing, and Deployment.






