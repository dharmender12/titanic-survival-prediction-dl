import os
import streamlit as st
import numpy as np
import pandas as pd

# Prevent TensorFlow from trying to reserve GPU memory in low-VRAM setups.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import pickle
## Title of the app
st.title("Titanic Survival Prediction App")

## Slider for Passenger Class
pclass = st.slider("Passenger Class (Pclass)", 1, 3, 1)

## Gender selection for selectbox
gender = st.selectbox("Enter the Passenger Gender",("male","female"))

## Slider for number of siblings/spouses aboard
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)

## Slider for number of parents/children aboard
parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)

## Fare Enter
fare = st.number_input("Enter the Fare paid by the passenger", min_value=0.0, step=0.1)

## Selectbox for port of embarkation
embarked = st.selectbox("Select the Port of Embarkation", ("Southampton", "Cherbourg", "Queenstown"))


## Dataframe to hold the user input
data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [gender],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

## Button to display the user input
# if st.button("Data"):
#     # st.write("User Input:")
#     st.write(data)


## Load the trained model
model = tf.keras.models.load_model('models/titanic_model.keras')

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

## Preprocess the user input
with open('models/onehot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

## Scale the numerical features
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## Encode the categorical features
data['Sex'] = label_encoder.transform(data['Sex'])
embarked = one_hot_encoder.transform(data[['Embarked']])

if hasattr(embarked, 'toarray'):
    embarked = embarked.toarray()

embarked_df = pd.DataFrame(embarked, columns=one_hot_encoder.get_feature_names_out())

data = pd.concat([data.drop('Embarked', axis=1), embarked_df], axis=1)

numeric_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
data[numeric_cols] = scaler.transform(data[numeric_cols])

model_feature_order = [
    'Pclass',
    'Sex',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked_Cherbourg',
    'Embarked_Queenstown',
    'Embarked_Southampton'
]
data = data.reindex(columns=model_feature_order, fill_value=0.0)

y = model.predict(data)
y = y[0][0]

def chance_of_survival(y):
    if y > 0.5:
        return f"The passenger will survive the journey"
    else:
        return f"The passenger will not survive the journey"

if st.button("Predict Survival Chance"):
    st.write(f"Probability of Passenger Survival Chance: {y:.2f}")
    st.write(chance_of_survival(y))

      