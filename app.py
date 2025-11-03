import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the trained model
model = tf.keras.models.load_model('Churn classification project/model.h5')

#load encoders and scaler
with open('Churn classification project/label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)

with open('Churn classification project/onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('Churn classification project/scaler.pkl','rb') as file:
    scaler=pickle.load(file)


#streamlit app
st.title('Customer Churn Prediction')

#User input
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0]) 
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance= st.number_input('Balance')
credit_score=st.slider('Credit Score',650,900)
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Product',1,4)
st.write('Yes -> 1 | No -> 0')
has_cr_card=st.selectbox('Has Credict Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One hot encode 'geography'
geo_encoder = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoder_df = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out()
)

# concatination 
input_data_df=pd.concat([input_data,geo_encoder_df],axis=1)
# print(input_data_df)

# Scale the input data
input_data_scaled=scaler.transform(input_data_df)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability :{prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.markdown(
        "<h3 style='color:red; font-size:28px;'>🚨 The customer is likely to churn.</h3>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<h3 style='color:green; font-size:28px;'>✅ The customer is not likely to churn.</h3>",
        unsafe_allow_html=True
    )
