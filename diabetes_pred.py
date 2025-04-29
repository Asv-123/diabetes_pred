import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('diabetes.csv')

st.set_page_config(page_title='Diabetes Dashboard', layout='wide')

st.title("Diabetes Dashboard")

st.subheader('Diabetes Statistics')

tab1,tab2,tab3,tab4 = st.tabs(['Glucose','BMI','Blood pressure','Diabetes Prediction'])

with tab1:
  st.subheader('Glucose analysis')

  c1,c2 = st.columns(2)

  with c1:
    gluc_data = data['Glucose']
    gluc_data = gluc_data.mean()
    gluc_data = round(gluc_data,1)
    c1.metric('Average glucose: ',gluc_data)

  with c2:
    fig_gluc = px.histogram(data,x = "Glucose",nbins= 30, color= 'Outcome' )
    fig_gluc.update_layout(title = 'Glucose Level Distribution',xaxis_title = "Glucose", yaxis_title = "Value")


    st.plotly_chart(fig_gluc)

with tab2:
  st.subheader('BMI')

  c1,c2 = st.columns(2)

  with c1:
    bmi = data['BMI']
    bmi = bmi.mean()
    bmi = round(bmi,1)
    c1.metric('Average BMI: ',bmi)
  with c2:
    fig_bmi = px.histogram(data, x = 'BMI', nbins= 30, color = 'Outcome')
    fig_bmi.update_layout(title = 'BMI Level Distribution', xaxis_title = 'BMI', yaxis_title = 'Value')

    st.plotly_chart(fig_bmi)

with tab3:
    st.subheader("Blood Pressure")

    c1,c2 = st.columns(2)

    with c1:
      bp = data['BloodPressure']
      bp = bp.mean()
      bp = round(bp,1)
      c1.metric("Average Blood Pressure: ",bp)
    with c2:
      fig_bp = px.histogram(data, x = 'BloodPressure',nbins = 30, color = 'Outcome')
      fig_bp.update_layout(title = 'Blood pressure Levels Distribution', xaxis_title = 'Blood Pressure', yaxis_title = 'Value')

      st.plotly_chart(fig_bp)

with tab4:
  st.subheader('Diabetes Prediction')

  x = data.drop('Outcome', axis=1)
  y = data['Outcome']

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  model = LogisticRegression()
  model.fit(x_train, y_train)


  gluc = st.number_input('Glucose Level: ', min_value=0)
  bp = st.number_input('Blood Pressure: ', min_value=0)
  skin = st.number_input('Skin Thickness', min_value=0)
  insulin = st.number_input('Insulin Level: ', min_value=0)
  bmi = st.number_input('BMI: ', min_value=0)
  dpf = st.number_input('Diabetes Pedigree Function: ')
  age = st.number_input('Age:', min_value=0)

  if st.button('Predict'):
    value = [[gluc, bp, skin, insulin, bmi, dpf, age]]
    prob = model.predict_proba(value)
    pred = model.predict(value)

    st.subheader('Prediction Result:')
    if prob[0][1] > 0.5:
      st.error('Result: Diabetic')
    else:
      st.success('Result: Non-Diabetic')

    st.write(f'Probability of being diabetic: {prob[0][1]}')



