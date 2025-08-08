#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model=joblib.load('XGB.pkl')

feature_names=[
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

st.title('计算器')

sepal_length = st.number_input('花萼长度:', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('花萼宽度:', min_value=0.0, max_value=10.0, value=5.0)
petal_length = st.number_input('花瓣长度:', min_value=0.0, max_value=10.0, value=5.0)
petal_width = st.number_input('花瓣宽度:', min_value=0.0, max_value=10.0, value=5.0)

feature_values=[sepal_length,sepal_width,petal_length,petal_width]
features=np.array([feature_values])

if st.button('Predict'):
    predicted_class=model.predict(features)[0]
    predicted_proba=model.predict_proba(features)[0]

    st.write(f'**Predicted Class:** {predicted_class} (1:Disease,0:No Disease)')
    st.write(f'**Prediction Probabilities:** {predicted_proba}')

    probability=predicted_proba[predicted_class] * 100
    if predicted_class==1:
        advice=(
            f'According to our model, you have a high risk of disease.'
            f'The model predicts that your probability of having disease is {probability:.1f}%.'
            'hhh.'
        )
    else:
        advice=(
            f'According to our model, you have a low risk of disease.'
            f'The model predicts that your probability of not having disease is {probability:.1f}%.'
            'ggg.'
        )

    st.write(advice)

    st.subheader('SHAP Force Plot Explanation')

    explainer_shap=shap.Explainer(model)
    shap_values=explainer_shap.shap_values(pd.DataFrame([feature_values],columns=feature_names))

    if predicted_class==1:
        shap.force_plot(explainer_shap.expected_value[1],shap_values[:,:,1],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0],shap_values[:,:,0],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)

    plt.savefig('shap_force_plot.png',bbox_inches='tight',dpi=1200)
    st.image('shap_force_plot.png',caption='SHAP Force Plot Explanation')


# In[ ]:




