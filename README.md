<h1>Heart Disease Prediction Project</h1>
This project focuses on predicting the likelihood of heart disease based on a diverse set of health features extracted from patient data.

You can access the heart disease predictor web application [here](https://heart-disease-prediction-xaavvcgce2wpmd4fay9wv4.streamlit.app/)
Enter your health information such as age, cholesterol, and blood pressure to get an instant prediction.

<h3>Objective</h3>
The goal of this project was to develop a machine learning model that can accurately predict whether a patient is at risk of heart disease. The model categorizes patients into two groups: at risk or not at risk based on key health metrics such as age, cholesterol levels, blood pressure, and other relevant features.

<h3>Approach</h3>
<h4>1. Data Exploration and Preprocessing</h4>
The project began with a thorough exploration and preprocessing of the dataset. This included handling missing values, encoding categorical variables, scaling numerical features, and ensuring data quality. Features were selected based on their relevance to heart disease prediction, such as:
<ul>
<li>Age</li>
<li>Resting Blood Pressure</li>
<li>Cholesterol</li>
<li>Max Heart Rate</li>
<li>Chest Pain Type</li>
<li>Resting ECG</li>
</ul>
<h4>2. Model Selection and Training</h4>
Various classification algorithms were applied and evaluated for performance on this binary classification task. These algorithms included:
<ul>
<li>Random Forest</li>
<li>Support Vector Machine</li>
<li>Logistic Regression</li>
<li>K-Nearest Neighbors</li>
<li>Decision Tree</li>
</ul>
Each model was tuned and tested to identify the best-performing one based on metrics like:
<ul>
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1 Score</li>
</ul>
Cross-validation was used to ensure the model's robustness and prevent overfitting.

<h4>3. Model Deployment</h4>

The final model, which was a Random Forest Classifier, was deployed as an interactive web application using Streamlit. The app allows users to input various health metrics and receive real-time predictions on whether they are at risk of heart disease.

<h4>4. Evaluation</h4>
The performance of the model was evaluated on a test dataset using the following metrics:

Accuracy: The overall correctness of predictions.
Precision: The proportion of positive identifications that were actually correct.
Recall: The ability of the model to identify all positive cases.
F1 Score: A balance between precision and recall.

<h5>5. Key Features</h5>
Interactive Web Application: The model has been deployed on a server using Streamlit, allowing users to easily input their health data and receive a heart disease risk prediction.
Cross-Validation: Ensured that the model generalizes well to unseen data.
Real-Time Prediction: The app provides instant results based on user inputs.

<h3>Conclusion</h3>
This end-to-end heart disease prediction project demonstrates the potential of machine learning in the healthcare domain. It highlights the importance of data preprocessing, algorithm selection, and evaluation, while providing a practical tool that could assist in early detection of heart disease.

<h3>Technologies Used</h3>
<ul>
<li>Python</li>
<li>Pandas, NumPy for data manipulation</li>
<li>Scikit-learn for machine learning models</li>
<li>Streamlit for deployment</li>
<li>Matplotlib, Seaborn for visualizations</li>
</ul>
