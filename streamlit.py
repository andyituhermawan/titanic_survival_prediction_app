import streamlit as st
import pandas as pd
import numpy as np
import pickle

#memberikan judul 
st.title('Survive Predictor')
st.write('This website can be used to predict survival rate Titanic Customer')

#menambahkan sidebar
st.sidebar.header("Please input customer's features")

#create user input
def create_user_input():
    #numerical : 'pclass', 'age', 'sibsp', 'parch', 'fare'
    pclass=st.sidebar.slider('pclass',min_value=1,max_value=3,value=1)
    age=st.sidebar.slider('age', min_value=1,max_value=80,value=20)
    sibsp=st.sidebar.slider('sibsp', min_value=0,max_value=8,value=1)
    parch=st.sidebar.slider('parch', min_value=0,max_value=6,value=1)
    fare=st.sidebar.number_input('fare',min_value=0.0,max_value=513.0,value=30.0)

    #categorical : 'sex' 'embarked'
    sex=st.sidebar.radio('sex',['male','female'])
    embarked=st.sidebar.radio('embarked',['S','C','Q'])

    #create dictionary from data input
    user_data={
        'pclass':pclass,
        'sex':sex,
        'age':age,
        'sibsp':sibsp,
        'parch':parch,
        'fare':fare,
        'embarked':embarked
    }

    #convert to dataframe
    user_data_df=pd.DataFrame([user_data])
    return user_data_df

#define customer data
data_customer=create_user_input()

#create 2 containers
col1, col2=st.columns(2)

#kiri
with col1:
    st.subheader('Customer Features')
    st.write(data_customer.transpose())
#load model
with open('best_model.sav','rb') as f:
    model_loaded=pickle.load(f)
#predict to customer data
target=model_loaded.predict(data_customer) #target 1 atau 0 dari model
probability=model_loaded.predict_proba(data_customer)[0] 

#menampilkan hasil prediksi
#kanan
with col2:
    st.subheader('Prediction Result')
    if target ==1:
        st.write('This customer will SURVIVE')
    else:
        st.write('This customer will NOT SURVIVE')
    #display probability
    st.write(f"Probability of survive : {probability[1]:.2f}")
        