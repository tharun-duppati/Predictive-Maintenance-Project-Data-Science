import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import re
import pickle

st.title('Machine Failure prediction using XG-Boost')

st.sidebar.header('User Input Parameters')

with open('std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

def user_input_features():
    Air_temperature = st.sidebar.number_input("Air temperature [K]", format='%0.1f', min_value=290.0, max_value=310.0, step=0.1)
    Process_temperature = st.sidebar.number_input("Process temperature [K]", format='%0.1f', min_value=300.0, max_value=320.0, step=0.1)
    Rotational_speed = st.sidebar.number_input("Rotational speed [rpm]", format='%i', min_value=1160, max_value=1900)
    Torque_Nm = st.sidebar.number_input("Torque [Nm]", format='%i', min_value=3, max_value=80)
    Tool_wear = st.sidebar.number_input("Tool wear [min]", format='%i', min_value=0, max_value=280)

    Type = st.sidebar.selectbox('Type',('H','L', 'M'))

    data = {'Air temperature [K]'    : [Air_temperature],
            'Process temperature [K]': [Process_temperature],
            'Rotational speed [rpm]' : [Rotational_speed],
            'Torque [Nm]'            : [Torque_Nm],
            'Tool wear [min]'        : [Tool_wear],
            'Type'                   : [Type]
            }
    st.subheader('User Input parameters')
    st.write(pd.DataFrame(data))

    data = {'Air temperature [K]'    : [Air_temperature],
            'Process temperature [K]': [Process_temperature],
            'Rotational speed [rpm]' : [Rotational_speed],
            'Torque [Nm]'            : [Torque_Nm],
            'Tool wear [min]'        : [Tool_wear],
            'Type_H': [1 if Type=='H' else 0],
            'Type_L': [1 if Type=='L' else 0],
            'Type_M': [1 if Type=='M' else 0]
            }
    data = pd.DataFrame(data)

    features = pd.DataFrame(data,index = [0])
    return features
    
df = user_input_features()

# st.write(df)

c = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[c] = scaler.transform(df[c])

df.columns = df.columns.str.replace(r"[\[\]<]", "", regex=True)

prediction = xgboost_model.predict(df)
prediction_proba = xgboost_model.predict_proba(df)

st.subheader('Predicted Result: '+'There is a chance of Machine failure' if prediction_proba[0][1] > 0.5 else 'Predicted Result: '+'Machine will not fail')
# st.write('There is a chance of Machine failure' if prediction_proba[0][1] > 0.5 else 'Machine will not fail')

st.subheader('Prediction Probability')

# Create a DataFrame for displaying probabilities
prob_df = pd.DataFrame(prediction_proba, columns=["Not Fail", "Fail"])

# Apply styling to highlight the cell with the highest probability
styled_df = prob_df.style.applymap(lambda val: 'background-color: yellow' if val == prob_df.max().max() else '', subset=pd.IndexSlice[:, ["Not Fail", "Fail"]])

# Display the styled DataFrame
st.dataframe(styled_df)