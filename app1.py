import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

# Load trained model
file = open('titanic_model.pkl', 'rb')  # Replace with your trained model file name
model = pickle.load(file)
file.close()


st.title("Titanic Survival Predictor")

# Collect user inputs
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, max_value=10, value=0)
fare = st.slider('Fare (in $)', 0.0, 512.3292, 15.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])  # C = Cherbourg, Q = Queenstown, S = Southampton

if st.button('Predict Survival'):

    # Preprocess user inputs
    sex = 1 if sex == 'male' else 0  # Assuming male=1 and female=0 in training
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    embarked = embarked_mapping[embarked]

    # Create query array
    query = np.array([[[1, 2, 3, 4, 5, 6, 7]]])
    query = np.squeeze(query, axis=1)
    
    # Feature names from the model
    columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    query_df = pd.DataFrame([query], columns=columns)
    
    # Predict survival
    prediction = model.fit_predict(query_df)  # Assuming model output is 0 or 1

    # Display result
    if prediction == 1:
        st.title("The passenger is likely to SURVIVE.")
    else:
        st.title("The passenger is unlikely to SURVIVE.")
