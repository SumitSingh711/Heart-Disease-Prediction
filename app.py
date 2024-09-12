import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('dataset.csv')

df.rename(columns={'chest pain type': 'chest_pain_type', 'resting bp s': 'resting_bp_s', 'fasting blood sugar': 'fasting_blood_sugar', 'resting ecg':'resting_ecg', 'max heart rate':'max_heart_rate', 'exercise angina':'exercise_angina', 'ST slope':'st_slope'}, inplace=True)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = joblib.load('pipeline.joblib')

column_list = df.columns.tolist()
X_columns = [col for col in column_list if col != 'target']

# Initialize log DataFrame
if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=X_columns)


# Define the function to log predictions
def log_prediction(input_data, prediction):
    new_entry = pd.DataFrame({
        'age': [input_data['age'].iloc[0]],
        'sex': [input_data['sex'].iloc[0]],
        'chest_pain_type': [input_data['chest_pain_type'].iloc[0]],
        'resting_bp_s': [input_data['resting_bp_s'].iloc[0]],
        'cholesterol': [input_data['cholesterol'].iloc[0]],
        'fasting_blood_sugar': [input_data['fasting_blood_sugar'].iloc[0]],
        'resting_ecg': [input_data['resting_ecg'].iloc[0]],
        'max_heart_rate': [input_data['max_heart_rate'].iloc[0]],
        'exercise_angina': [input_data['exercise_angina'].iloc[0]],
        'oldpeak': [input_data['oldpeak'].iloc[0]],
        'st_slope': [input_data['st_slope'].iloc[0]],
        'Prediction': ['Heart Disease' if prediction == 1 else 'No Heart Disease']
    })
    st.session_state.log_df = pd.concat([st.session_state.log_df, new_entry], ignore_index=True)


# web application
st.set_page_config(
    page_title='Heart Disease Predictor',
    page_icon='🧑🏻‍⚕️',
)

st.title(" 📱 Heart Disease Predictor")

st.subheader("Welcome to the Heart Disease Predictor App")
st.write("""Here, you can assess your risk of heart disease based on your health metrics""")


st.subheader("About")
st.info("This application predicts the heart disease by providing details such as age, cholesterol levels, and other key health indicators,"
        "our app uses advanced algorithms to predict your likelihood of heart disease. "
        "Get personalized insights and take proactive steps towards better heart health.")


st.subheader("Input Features")

age = st.number_input(
    "**Age** *(Years)*",
    min_value=1,  # Minimum year
    max_value=100,  # Maximum year
    value=40,      # Default value
    step=5         # Step value
)

sex = st.selectbox(
    '**Sex** *(Male=1 or Female=0)*',
    options=X_train.sex.unique()
)

chest_pain_type = st.selectbox(
    "**Chest_Pain_Type** *(typical angina=1, atypical angina=2, non-anginal pain=3, asymptomatic=4)*",
    options=X_train.chest_pain_type.unique()
)

resting_bp_s = st.number_input(
    '**Resting blood pressure** *(mm Hg)*',
    min_value=0,  # Minimum year
    max_value=250,  # Maximum year
    value=50,  # Default value
    step=20  # Step value
)

cholesterol = st.number_input(
    "**Serum cholesterol** *(mg/dl)*",
    min_value=0,  # Minimum year
    max_value=700,  # Maximum year
    value=200,      # Default value
    step=20       # Step value
)

fasting_blood_sugar = st.selectbox(
     '**Fasting blood sugar** *(sugar > 120mg/dL=1, sugar < 120mg/dL=0)*',
    options=X_train.fasting_blood_sugar.unique()
)

resting_ecg = st.selectbox(
    '**Resting electrocardiogram results** *(normal=0, ST-T wave abnormality (T wave inversions and/or ST elevation/depression of > 0.05 mV)=1, Probable or Definite Left Ventricular hypertrophy by Estes’ criteria=2)*',
    options=X_train.resting_ecg.unique()
)

max_heart_rate = st.number_input(
    "**Maximum heart rate** *(bpm)*",
    min_value=40,  # Minimum year
    max_value=220,  # Maximum year
    value=80,      # Default value
    step=10      # Step value
)

exercise_angina = st.selectbox(
     '**Exercise induced angina** *(Yes=1 or No=0)*',
     options=X_train.exercise_angina.unique()
)

oldpeak = st.number_input(
     "**Oldpeak=ST** *(depression)*",
    min_value=0.0,  # Minimum year
    max_value=10.0,  # Maximum year
    value=5.0,      # Default value
    step=2.0      # Step value
)

st_slope = st.selectbox(
     '**The slope of the peak exercise ST segment** *(Upward=1, Flat=2, Downward=3)*',
     options=X_train.st_slope.unique()
)


X_new = pd.DataFrame(dict(
	age = [age],
	sex = [sex],
	chest_pain_type	= [chest_pain_type],
	resting_bp_s = [resting_bp_s],
	cholesterol = [cholesterol],
    fasting_blood_sugar = [fasting_blood_sugar],
	resting_ecg = [resting_ecg],
	max_heart_rate=[max_heart_rate],
	exercise_angina=[exercise_angina],
	oldpeak=[oldpeak],
    st_slope = [st_slope]
))


if st.button('Predict Patient Health'):

    prediction = pipeline.predict(X_new)[0]
    # cost_category = target_mapping[pred[0]]

    if prediction==1:
        st.markdown("<h4 style='color: red;'>Prediction: The patient is likely to have heart disease.</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='color: green;'>Prediction: The patient is healthy.</h4>", unsafe_allow_html=True)

    # Log the prediction
    log_prediction(X_new, prediction)

st.write(X_new)

st.subheader("Prediction Log History")
st.dataframe(st.session_state.log_df)